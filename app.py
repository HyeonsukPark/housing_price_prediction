import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model

@st.cache_resource
def load_model():
    ml_model = joblib.load('rf_model.joblib')
    return ml_model

model = load_model()


# Set up the Streamlit page
st.set_page_config(
    page_title = "Lisbon Housing Price Prediction Dashboard",
    layout="wide",
)

st.title("Housing Price Prediction Dashboard")
st.write("This dashboard predicts housing prices in Lisbon, using a trained Random Forest model.")

# --- Side bar ---#

st.sidebar.header("Input House Features")

# Create input widgets
def user_input_features():

    district = st.sidebar.selectbox(
        "Choose a district",
        ("Lisboa")
    )

    city = st.sidebar.selectbox(
        "Choose a city",
        ("Sintra", "Lisboa", "Azambuja", "Cascais", "Vila Franca de Xira", "Odivelas",
         "Loures", "Lourinhã", "Alenquer", "Oeiras", "Amadora", "Arruda dos Vinhos", "Torres Vedras",
         "Mafra", "Sobral de Monte Agraço", "Cadaval")
    )

    town = st.sidebar.selectbox(
        "Choose a Town",
        (   "Algueirão-Mem Martins", "Misericórdia", "Ajuda", "Azambuja", "Casal de Cambra", "Carcavelos e Parede", "São Domingos de Rana", "Alcabideche",
            "Alverca do Ribatejo e Sobralinho", "Cacém e São Marcos", "Santo António", "Cascais e Estoril", "Santa Maria Maior",
            "São Domingos de Benfica", "Odivelas", "Santa Iria de Azoia", "São João da Talha e Bobadela", "Loures", "Lumiar",
            "Massamá e Monte Abraão", "Vimeiro", "Aldeia Galega da Merceana e Aldeia Gavinha", "Avenidas Novas", "Carregado e Cadafais",
            "Sacavém e Prior Velho", "Campolide", "Vialonga", "Alvalade", "Moscavide e Portela", "Oeiras e São Julião da Barra",
            "Paço de Arcos e Caxias", "Encosta do Sol", "Beato", "Arroios", "Carnaxide e Queijas", "Agualva e Mira-Sintra",
            "Campo de Ourique", "Pontinha e Famões", "Barcarena", "Póvoa de Santa Iria e Forte da Casa", "Venteira", "Algés",
            "Linda-a-Velha e Cruz Quebrada-Dafundo", "Parque das Nações", "Arruda dos Vinhos", "Ramada e Caneças", "Colares",
            "Rio de Mouro", "Santo António dos Cavaleiros e Frielas", "Silveira", "Sintra (Santa Maria e São Miguel)",
            "São Martinho e São Pedro de Penaferrim)", "Campelos e Outeiro da Cabeça", "Belém", "São Vicente", "Lourinhã e Atalaia",
            "A dos Cunhados e Maceira", "Estrela", "Ventosa", "Queluz e Belas", "Carnide", "Alenquer (Santo Estêvão e Triana)",
            "Alhandra", "São João dos Montes e Calhandriz", "Venda do Pinheiro e Santo Estêvão das Galés", "São Quintino", "Benfica",
            "Areeiro", "Olivais", "Bucelas", "Almargem do Bispo", "Pêro Pinheiro e Montelavar", "Águas Livres", "Alcântara", "Carnota",
            "Santa Clara", "Ericeira", "Mina de Água", "São João das Lampas e Terrugem", "Camarate", "Unhos e Apelação", "Vilar",
            "Santo Isidoro", "São Pedro da Cadeira", "Porto Salvo", "Penha de França", "Falagueira - Venda Nova", "Azueira e Sobral da Abelheira",
            "Malveira e São Miguel de Alcainça", "Marvila", "Sobral de Monte Agraço", "Vila Franca de Xira", "Santo Antão e São Julião do Tojal",
            "Torres Vedras (São Pedro, Santiago, Santa Maria do Castelo e São Miguel) e Matacães",
            "Lamas e Cercal", "Santa Bárbara", "Póvoa de Santo Adrião e Olival Basto", "Sapataria", "Painho e Figueiros",
            "Castanheira do Ribatejo e Cachoeiras", "Alguber", "Miragaia e Marteleira", "Aveiras de Cima", "Ribamar", "Enxara do Bispo",
            "Gradil e Vila Franca do Rosário", "Milharado", "Manique do Intendente", "Vila Nova de São Pedro e Maçussa",
            "Fanhões", "Mafra", "Cadaval e Pêro Moniz", "Encarnação", "Carvoeira", "Alfragide", "Ota", "Vermelha", "Abrigada e Cabanas de Torres",
            "Aveiras de Baixo", "Maxial e Monte Redondo", "Ramalhal", "Santiago dos Velhos", "Alcoentre", "Freiria", "Vila Verde dos Francos",
            "Carvoeira e Carmões", "Olhalvo", "Ribafria e Pereiro de Palhacana", "São Bartolomeu dos Galegos e Moledo",
            "Dois Portos e Runa", "Ponte do Rol", "Peral", "Igreja Nova e Cheleiros", "Lousa", "Arranhó", "Vila Nova da Rainha",
            "Meca", "Reguengo Grande", "Moita dos Ferreiros", "Turcifal", "Vale do Paraíso", "Cardosas"
        )
    )

    housing_type = st.sidebar.selectbox(
        "Choose a housing type",
        ("Apartment", "House", "Duplex", "Other - Residential", "Studio", "Manor", "Mansion")
    )

    energy_certificate = st.sidebar.selectbox(
        "Does it have a certificate of energy?",
        ("No Certificate", "A+", "A", "B", "B-", "C", "D", "E", "F", "G", "NC")
    )

    total_area = st.sidebar.slider("Total Housing Area", 0, 367)

    has_parking = st.sidebar.selectbox(
        "Does it have a parking?",
        ("Yes", "No")
    )

    parking = st.sidebar.selectbox(
        "How many parkings?",
        ("0", "1", "2", "3")
    )

    floor = st.sidebar.selectbox(
        "How many floors?",
        ("1st Floor", "5th Floor", "Basement Level", "3rd Floor", "2nd Floor", "Duplex", "4th Floor",
         "Ground Floor", "7th Floor", "6th Floor", "Above 10th Floor", "8th Floor", "9th Floor"
         , "Attic", "Triplex", "Basement", "Top Floor", "Service Floor")
    )

    construction_year = st.sidebar.slider("Construction Year",  1900, 2030)

    energy_efficiency_level = st.sidebar.selectbox(
        "What is the level of energy efficiency?",
        ("No Certificate", "A+", "A", "B", "B-", "C", "D", "E", "F", "G", "NC")
    )

    garage = st.sidebar.selectbox(
        "Is there a garage?",
        ("Yes", "No")
    )

    elevator = st.sidebar.selectbox(
        "Is there a elevator?",
        ("Yes", "No")
    )

    electric_cars_charging = st.sidebar.selectbox(
        "Is there a charging for electric cars?",
        ("Yes", "No")
    )

    total_rooms = st.sidebar.slider("Total rooms", 0, 10)

    number_of_wc = st.sidebar.slider("Number of WC", 0, 10)

    number_of_bathrooms = st.sidebar.slider("Number of Bathrooms", 0, 10)

    # Create a dictionary of the input features
    data = {
        'District': district,
        'City': city,
        'Town': town,
        'Type': housing_type,
        'EnergyCertificate': energy_certificate,
        'TotalArea': total_area,
        'HasParking': has_parking,
        'Parking': parking,
        'Floor': floor,
        'ConstructionYear': construction_year,
        'EnergyEfficiencyLevel': energy_efficiency_level,
        'Garage': garage,
        'Elevator': elevator,
        'ElectricCarsCharging': electric_cars_charging,
        'TotalRooms': total_rooms,
        'NumberOfWC': number_of_wc,
        'NumberOfBathrooms': number_of_bathrooms,
    }

    features = pd.DataFrame(data, index=[0])

    return features


df_input = user_input_features()

# --- Main Page Content ---

st.subheader("User Input Features")
st.write(df_input)

# --- Prediction Section ---
st.subheader("Prediction")

# Make a prediction when a button is clicked

if st.sidebar.button("Predict Price"):

    prediction = model.predict(df_input)

    # display the result
    st.success(f"The predicted housing price is: **€ {prediction[0]:,.2f}**")





