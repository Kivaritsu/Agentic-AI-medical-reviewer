import os
from typing import List, Dict, Any
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import (
    FileReadTool, 
    DirectoryReadTool, 
    PDFSearchTool, 
    DOCXSearchTool,
    TXTSearchTool,
    JSONSearchTool,
    CSVSearchTool,
    SerperDevTool,
    FileWriterTool
)

# Set up environment variables
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"
os.environ["SERPER_API_KEY"] = "your-serper-api-key-here"

@CrewBase
class MedicalAnalysisCrew:
    """Medical Analysis Crew for comprehensive patient data processing"""
    
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    
    def __init__(self):
        # Initialize tools for document processing
        self.file_tools = [
            FileReadTool(),
            DirectoryReadTool(directory='./medical_records'),
            PDFSearchTool(),
            DOCXSearchTool(),
            TXTSearchTool(),
            JSONSearchTool(),
            CSVSearchTool()
        ]
        
        # Initialize web search and analysis tools
        self.analysis_tools = [
            SerperDevTool(),
            FileWriterTool()
        ]

    @agent
    def medical_records_analyzer(self) -> Agent:
        """Agent responsible for reading and processing medical documents"""
        return Agent(
            config=self.agents_config['medical_records_analyzer'],
            tools=self.file_tools,
            verbose=True,
            memory=True,
            max_execution_time=300,
            allow_delegation=False
        )
    
    @agent  
    def clinical_summary_specialist(self) -> Agent:
        """Agent responsible for clinical analysis and medication assessment"""
        return Agent(
            config=self.agents_config['clinical_summary_specialist'],
            tools=self.analysis_tools + [FileWriterTool()],
            verbose=True,
            memory=True,
            max_execution_time=400,
            allow_delegation=False
        )
    
    @agent
    def predictive_health_analyst(self) -> Agent:
        """Agent responsible for predictive health risk assessment"""
        return Agent(
            config=self.agents_config['predictive_health_analyst'],
            tools=self.analysis_tools + [FileWriterTool()],
            verbose=True,
            memory=True,
            max_execution_time=500,
            allow_delegation=False
        )
    
    @task
    def medical_document_processing_task(self) -> Task:
        """Task for processing and extracting medical document data"""
        return Task(
            config=self.tasks_config['medical_document_processing'],
            agent=self.medical_records_analyzer(),
            output_file='extracted_medical_data.json'
        )
    
    @task
    def clinical_medication_analysis_task(self) -> Task:
        """Task for clinical analysis and medication assessment"""
        return Task(
            config=self.tasks_config['clinical_medication_analysis'],
            agent=self.clinical_summary_specialist(),
            context=[self.medical_document_processing_task()],
            output_file='clinical_analysis_report.pdf'
        )
    
    @task
    def predictive_risk_assessment_task(self) -> Task:
        """Task for predictive health risk assessment"""
        return Task(
            config=self.tasks_config['predictive_risk_assessment'],
            agent=self.predictive_health_analyst(),
            context=[self.medical_document_processing_task(), self.clinical_medication_analysis_task()],
            output_file='risk_assessment_report.pdf'
        )
    
    @task
    def comprehensive_report_generation_task(self) -> Task:
        """Task for generating comprehensive medical summary report"""
        return Task(
            config=self.tasks_config['comprehensive_report_generation'],
            agent=self.clinical_summary_specialist(),
            context=[
                self.medical_document_processing_task(),
                self.clinical_medication_analysis_task(),
                self.predictive_risk_assessment_task()
            ],
            output_file='comprehensive_medical_summary.pdf'
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates and configures the medical analysis crew"""
        return Crew(
            agents=[
                self.medical_records_analyzer(),
                self.clinical_summary_specialist(), 
                self.predictive_health_analyst()
            ],
            tasks=[
                self.medical_document_processing_task(),
                self.clinical_medication_analysis_task(),
                self.predictive_risk_assessment_task(),
                self.comprehensive_report_generation_task()
            ],
            process=Process.sequential,
            verbose=True,
            memory=True,
            cache=True,
            max_rpm=10,
            share_crew=True,
            planning=True,
            step_callback=self._step_callback,
            task_callback=self._task_callback
        )

    def _step_callback(self, step_output):
        """Callback function for monitoring agent steps"""
        print(f"Step completed: {step_output}")
        
    def _task_callback(self, task_output):
        """Callback function for monitoring task completion"""
        print(f"Task completed: {task_output}")

    def run_analysis(self, patient_id: str = None, medical_records_path: str = "./medical_records") -> Dict[str, Any]:
        """
        Run the complete medical analysis process
        
        Args:
            patient_id: Unique identifier for the patient
            medical_records_path: Path to directory containing medical records
            
        Returns:
            Dict containing analysis results and file paths
        """
        
        inputs = {
            'patient_id': patient_id or 'default_patient',
            'medical_records_path': medical_records_path,
            'analysis_date': '2025-08-20',
            'requesting_physician': 'System Analysis',
            'analysis_type': 'comprehensive_medical_review'
        }
        
        print(f"Starting medical analysis for patient: {inputs['patient_id']}")
        print(f"Processing medical records from: {medical_records_path}")
        
        try:
            # Execute the crew
            result = self.crew().kickoff(inputs=inputs)
            
            print("Medical analysis completed successfully!")
            
            return {
                'status': 'completed',
                'patient_id': inputs['patient_id'],
                'analysis_results': result,
                'output_files': [
                    'extracted_medical_data.json',
                    'clinical_analysis_report.pdf', 
                    'risk_assessment_report.pdf',
                    'comprehensive_medical_summary.pdf'
                ],
                'completion_time': inputs['analysis_date']
            }
            
        except Exception as e:
            print(f"Error during medical analysis: {str(e)}")
            return {
                'status': 'error',
                'patient_id': inputs['patient_id'],
                'error_message': str(e),
                'completion_time': inputs['analysis_date']
            }

# Custom Tools for Medical Analysis
class DrugInteractionTool:
    """Custom tool for checking drug-drug interactions"""
    
    def __init__(self):
        self.name = "drug_interaction_checker"
        self.description = "Check for drug-drug interactions and adverse effects"
    
    def run(self, medications: List[str]) -> Dict[str, Any]:
        """
        Check for interactions between medications
        
        Args:
            medications: List of medication names
            
        Returns:
            Dict containing interaction analysis
        """
        # Implementation would connect to drug interaction databases
        # like DDInter, DrugBank, or clinical decision support APIs
        
        return {
            'interactions_found': len(medications) > 1,
            'high_risk_interactions': [],
            'moderate_risk_interactions': [],
            'monitoring_required': [],
            'recommendations': []
        }

class MedicalMLPredictor:
    """Custom ML tool for medical predictions"""
    
    def __init__(self):
        self.name = "medical_ml_predictor"
        self.description = "Apply ML algorithms for disease prediction and risk assessment"
    
    def predict_disease_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict disease risks using ML models
        
        Args:
            patient_data: Structured patient information
            
        Returns:
            Dict containing risk predictions
        """
        # Implementation would use trained ML models for:
        # - Cardiovascular risk prediction
        # - Diabetes risk assessment  
        # - Cancer screening recommendations
        # - Hospital readmission risk
        
        return {
            'cardiovascular_risk': {'score': 0.15, 'risk_level': 'moderate'},
            'diabetes_risk': {'score': 0.08, 'risk_level': 'low'},
            'cancer_screening_due': ['colonoscopy', 'mammography'],
            'readmission_risk': {'score': 0.12, 'risk_level': 'low'}
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize the medical analysis crew
    medical_crew = MedicalAnalysisCrew()
    
    # Example: Process medical records for a patient
    result = medical_crew.run_analysis(
        patient_id="PATIENT_001",
        medical_records_path="./sample_medical_records"
    )
    
    print("\n=== MEDICAL ANALYSIS RESULTS ===")
    print(f"Status: {result['status']}")
    print(f"Patient ID: {result['patient_id']}")
    
    if result['status'] == 'completed':
        print(f"Generated Files: {', '.join(result['output_files'])}")
        print("Analysis completed successfully!")
    else:
        print(f"Error: {result['error_message']}")

# Configuration for deployment
DEPLOYMENT_CONFIG = {
    'environment': 'production',
    'security_level': 'hipaa_compliant',
    'data_retention_days': 365,
    'backup_enabled': True,
    'audit_logging': True,
    'encryption_enabled': True,
    'access_control': 'role_based'
}