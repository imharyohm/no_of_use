import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5ForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

# Set the model name
model_name = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Load CSV files for each domain (adjust file paths as needed)
symptoms_df = pd.read_csv("symtp.csv")                # Columns: disease_name, symptoms
descriptions_df = pd.read_csv("descriptions.csv")        # Columns: disease_name, description
precautions_df = pd.read_csv("precautions.csv")          # Columns: disease_name, precautions
medications_df = pd.read_csv("medications.csv")          # Columns: disease_name, medications
workouts_df = pd.read_csv("workout.csv") # Columns: disease_name, workout_recommendations

print("Symptoms columns:", symptoms_df.columns)
print("Descriptions columns:", descriptions_df.columns)
print("Precautions columns:", precautions_df.columns)
print("Medications columns:", medications_df.columns)
print("Workout Recommendations columns:", workouts_df.columns)

symptoms_df.columns = symptoms_df.columns.str.strip()
descriptions_df.columns = descriptions_df.columns.str.strip()
precautions_df.columns = precautions_df.columns.str.strip()
medications_df.columns = medications_df.columns.str.strip()
workouts_df.columns = workouts_df.columns.str.strip()

# Merge the DataFrames on 'disease_name'
df = symptoms_df.merge(descriptions_df, on='disease_name', how='left') \
                .merge(precautions_df, on='disease_name', how='left') \
                .merge(medications_df, on='disease_name', how='left') \
                .merge(workouts_df, on='disease_name', how='left')


print("Merged DataFrame:")
print(df.head())

# 3. Create Disease-to-Label Mapping
# --------------------------
unique_diseases = sorted(df["disease_name"].unique())
disease_to_label = {disease: idx for idx, disease in enumerate(unique_diseases)}
print("Disease to label mapping:", disease_to_label)

# Define a custom dataset class

class DiseaseDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer, max_length=512):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Define columns that are not symptoms.
        self.non_symptom_cols = [
            'disease_name', 'description', 'precautions',
            'medications', 'workout_recommendations'
        ]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # If there's a dedicated "symptoms" column, use it; otherwise, combine all other columns.
        if 'symptoms' in self.data.columns:
            symptoms = row['symptoms']
        else:
            symptom_cols = [col for col in self.data.columns if col not in self.non_symptom_cols]
            symptoms = ', '.join([str(row[col]) for col in symptom_cols if pd.notnull(row[col])])
        
        # Helper to safely convert any field to a string.
        def safe_str(val):
            return str(val) if pd.notnull(val) else ""
        
        symptoms = safe_str(symptoms)
        disease_name = safe_str(row['disease_name'])
        description = safe_str(row['description'])
        precautions = safe_str(row['precautions'])
        medications = safe_str(row['medications'])
        workout_recommendations = safe_str(row['workout_recommendations'])
        
        # Tokenize inputs and outputs.
        inputs = self.tokenizer(
            symptoms, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length
        )
        
        description_output = self.tokenizer(
            description, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length
        )
        
        precautions_output = self.tokenizer(
            precautions, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length
        )
        
        medications_output = self.tokenizer(
            medications, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length
        )
        
        workout_output = self.tokenizer(
            workout_recommendations, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length
        )
        
        # Map disease_name to its label index
        disease_name_label = disease_to_label[disease_name]
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels_disease_name': torch.tensor(disease_name_label, dtype=torch.long),  # Single integer label
            'labels_description': description_output['input_ids'].squeeze(),
            'labels_precautions': precautions_output['input_ids'].squeeze(),
            'labels_medications': medications_output['input_ids'].squeeze(),
            'labels_workout': workout_output['input_ids'].squeeze(),
        }

# Split dataset into training and validation sets
train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = DiseaseDataset(train_data, tokenizer)
val_dataset = DiseaseDataset(val_data, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2)

# Define a multitask model with T5 for text generation and a T5-based classification head for disease names.

class DiseaseModel(torch.nn.Module):
    def __init__(self):
        super(DiseaseModel, self).__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        # Dynamically set num_labels to the number of unique diseases
        num_labels = len(disease_to_label)  # Number of unique diseases
        self.classifier = T5ForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    def forward(self, input_ids, attention_mask, labels_disease_name, labels_description, labels_precautions, labels_medications, labels_workout):
        # Classification output for disease name prediction
        disease_name_output = self.classifier(input_ids, attention_mask=attention_mask, labels=labels_disease_name)

        # Generation outputs for the additional domains
        description_output = self.t5(input_ids=input_ids, attention_mask=attention_mask, labels=labels_description)
        precautions_output = self.t5(input_ids=input_ids, attention_mask=attention_mask, labels=labels_precautions)
        medications_output = self.t5(input_ids=input_ids, attention_mask=attention_mask, labels=labels_medications)
        workout_output     = self.t5(input_ids=input_ids, attention_mask=attention_mask, labels=labels_workout)
        
        return (disease_name_output.loss, description_output.loss,
                precautions_output.loss, medications_output.loss,
                workout_output.loss)


# Initialize the model and optimizer
model = DiseaseModel()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# Set device and move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

from tqdm import tqdm  # Add this import

# Training loop
epochs = 1
for epoch in range(epochs):
    model.train()
    total_loss = 0
    print(f"Epoch {epoch + 1}/{epochs}")
    
    # Use tqdm to create a progress bar for the training dataloader
    progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}", leave=False)
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_disease_name = batch['labels_disease_name'].to(device)
        labels_description = batch['labels_description'].to(device)
        labels_precautions = batch['labels_precautions'].to(device)
        labels_medications = batch['labels_medications'].to(device)
        labels_workout = batch['labels_workout'].to(device)
        
        optimizer.zero_grad()
        losses = model(input_ids, attention_mask,
                       labels_disease_name, labels_description,
                       labels_precautions, labels_medications, labels_workout)
        loss_value = sum(losses)  # Sum all loss components
        loss_value.backward()
        optimizer.step()
        total_loss += loss_value.item()
        
        # Update the progress bar with the current loss
        progress_bar.set_postfix({"Batch Loss": loss_value.item()})
    
    print(f"Epoch {epoch + 1}/{epochs}, Total Loss: {total_loss}")

# Evaluation on validation data
model.eval()
with torch.no_grad():
    total_val_loss = 0
    progress_bar = tqdm(val_dataloader, desc="Validation", leave=False)
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_disease_name = batch['labels_disease_name'].to(device)
        labels_description = batch['labels_description'].to(device)
        labels_precautions = batch['labels_precautions'].to(device)
        labels_medications = batch['labels_medications'].to(device)
        labels_workout = batch['labels_workout'].to(device)
        
        losses = model(input_ids, attention_mask,
                       labels_disease_name, labels_description,
                       labels_precautions, labels_medications, labels_workout)
        total_val_loss += sum(losses).item()
        progress_bar.set_postfix({"Validation Loss": total_val_loss})

    print(f"Validation Loss: {total_val_loss}")