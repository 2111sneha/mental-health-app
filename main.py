import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Feild

model = joblib.load('Mental_Health_Model.pkl')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
   


#A first pydantic model 
class StudentData(BaseModel):
    Age                    : int = Feild(..., ge=10, le=100, description="Age of the student (between 10 and 100)")
    Gender                 : Literal['Male', 'Female']
    Country                : str
    Academic_Level         : Literal['High School', 'Undergraduate', 'Graduate']
    Most_Used_Platform     : Literal['Facebook', 'Instagram', 'Twitter', 'Snapchat', 'TikTok', 'YouTube', 'LinkedIn','KakaoTalk', 'WeChat', 'Other']
    Purpose_Of_Use         : Literal['Socializing', 'Entertainment', 'Education', 'Work', 'Other']
    Avg_Daily_Usage_Hours  : float = Feild(..., ge=6, le=24, description="Average daily usage hours (between 6 and 24)")
    Daily_Unlocks          : int = Feild(..., ge=0)
    Study_Hours            : float = Feild(..., ge=0, le=24, description="Study hours (between 0 and 24)")
    Physical_Activity_Hours: float = Feild(..., ge=0, le=24, description="Physical activity hours (between 0 and 24)")
    Sleep_Hours_Per_Night  : float = Feild(..., ge=0, le=24, description="Sleep hours per night (between 0 and 24)")
    Stress_Level           : Literal['Low', 'Medium', 'High']
   

   class PredictionResponse(BaseModel):
       predicted_mental_health_score: float


@app.get("/")
def greet():
    return {'Welcome to Sneha"s AI School Guys'}


top_countries = ['Other','India','USA','Canada','Australia','UK','Germany','Mexico','Turkey','France']

@app.post('/predict', response_model=PredictionResponse)
def predict(data: StudentData):

   country_group = data.country if data.country in top_countries else "Other"
   
    input_row = pd.DataFrame([{
    'Age': data.Age,
    'Gender': data.Gender,
    'Country': data.Country,
    'Academic_Level': data.Academic_Level,
    'Most_Used_Platform': data.Most_Used_Platform,
    'Purpose_Of_Use': data.Purpose_Of_Use,
    'Avg_Daily_Usage_Hours': data.Avg_Daily_Usage_Hours,
    'Daily_Unlocks': data.Daily_Unlocks,
    'Study_Hours': data.Study_Hours,
    'Physical_Activity_Hours': data.Physical_Activity_Hours,
    'Sleep_Hours_Per_Night': data.Sleep_Hours_Per_Night,
    'Stress_Level': data.Stress_Level,
    'Grouped_country': country_group
    }])

    prediction = model.predict(input_row)[0]
    return PredictionResponse(predicted_mental_health_score=prediction)