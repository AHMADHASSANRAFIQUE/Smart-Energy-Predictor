import requests
import os
from datetime import datetime

currency_symbol = "Rs"

appliance_options = {
    "Fridge": {"Non-Inverter": 200, "Inverter": 150},
    "Washing Machine": {"Non-Inverter": 500, "Inverter": 400},
    "AC": {"Non-Inverter": 1500, "Inverter": 1000},
    "Fan": {"Non-Inverter": 75, "Inverter": 50},
    "TV": {"Non-Inverter": 100, "Inverter": 80},
    "Heater": {"Non-Inverter": 2000, "Inverter": 1500},
    "Water Heater": {"Non-Inverter": 3000, "Inverter": 2000},
    "LED Light": {"Non-Inverter": 10, "Inverter": 10},
    "Bulb": {"Non-Inverter": 60, "Inverter": 60}
}

def fetch_weather_forecast(city):
    try:        
        api_key = os.environ.get("OPENWEATHER_API_KEY")
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)
        data = response.json()
        if data.get("cod") != "200":
            return None
        forecast = []
        for entry in data["list"][:8]:  # Next 24 hours (8 entries, 3-hour intervals)
            forecast.append({
                "time": datetime.fromtimestamp(entry["dt"]).strftime("%Y-%m-%d %H:%M"),
                "temp": entry["main"]["temp"],
                "humidity": entry["main"]["humidity"]
            })
        return forecast
    except Exception:
        return None

def calculate_slab_bill(units):
    slabs = {
        "0-50": 5.0,
        "51-100": 10.0,
        "101-200": 15.0,
    }
    flat_tariff_above_200 = 35.0

    if units > 200:
        bill = units * flat_tariff_above_200
        slab_breakdown = {
            "Flat Tariff (>200)": (units, bill)
        }
        return bill, slab_breakdown

    slab_breakdown = {}
    remaining_units = units
    total_bill = 0.0

    if remaining_units > 0:
        units_in_slab = min(remaining_units, 50)
        cost = units_in_slab * slabs["0-50"]
        total_bill += cost
        slab_breakdown["0-50"] = (units_in_slab, cost)
        remaining_units -= units_in_slab

    if remaining_units > 0:
        units_in_slab = min(remaining_units, 50)
        cost = units_in_slab * slabs["51-100"]
        total_bill += cost
        slab_breakdown["51-100"] = (units_in_slab, cost)
        remaining_units -= units_in_slab

    if remaining_units > 0:
        units_in_slab = min(remaining_units, 100)
        cost = units_in_slab * slabs["101-200"]
        total_bill += cost
        slab_breakdown["101-200"] = (units_in_slab, cost)
        remaining_units -= units_in_slab

    return total_bill, slab_breakdown
