import pandas as pd
import json
import re
import easyocr
from PIL import Image
import cv2
import numpy as np
from io import BytesIO
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize OCR reader
reader = easyocr.Reader(["en"])

# Initialize session state correctly
session_state = {
    "processed": False,
    "df_store": pd.DataFrame(),
    "uploaded_image": None
}

# OCR Function
def OCR(image):
    """Extracts text from the uploaded image using EasyOCR."""
    image = Image.open(image).convert("RGB")  # Ensure RGB format
    image = np.array(image)  # Convert PIL image to NumPy array

    # Check if the image has multiple channels before converting to grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:  
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert only if RGB
    else:
        gray = image  # If already grayscale, use as is

    # Run OCR on the processed image
    results = reader.readtext(gray)
    extracted_text = [text[1] for text in results]  # Extract only text part
    return extracted_text

# Gemini LLM Response Function
def llm_response(extracted_text):
    """Sends extracted text to Gemini API for structured data extraction."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key='AIzaSyCcGqDgr0okwX3zivih9YRXciiCEixQx1c')
    
    prompt = f"""
    THE EXTRACTED TEXT IS FROM AN INVOICE.
    Extract the following structured data:
    - INVOICE (Invoice number)
    - DATE CREATED
    - VENDOR (Company name)
    - SALE TYPE
    - DELIVER TO
    - DESCRIPTION (as a list)
    - QUANTITY (Extract as an integer)
    - UNIT PRICE (Single item price)
    - EXTD PRICE or TOTAL (Total price)

    Return the data in **valid JSON format**.

    OCR Text:
    {extracted_text}
    """

    response = llm.invoke(prompt)
    return response.content  # Extracts actual text

# Convert LLM Response to DataFrame
def into_df(response):
    """Extracts JSON from response and appends it to a global DataFrame."""
    # Extract JSON from Response
    json_match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
    json_string = json_match.group(1) if json_match else response

    # Convert JSON to DataFrame
    try:
        extracted_json = json.loads(json_string)
    except json.JSONDecodeError:
        print("❌ Error: Invalid JSON format")
        return

    rows = []
    max_length = max([len(v) if isinstance(v, list) else 1 for v in extracted_json.values()])  

    for i in range(max_length):
        row = {key: (value[i] if isinstance(value, list) and i < len(value) else value)
               for key, value in extracted_json.items()}
        rows.append(row)

    new_df = pd.DataFrame(rows)

    # Filter out rows where "UNIT PRICE" is NaN or empty
    new_df = new_df[new_df["UNIT PRICE"].notna() & (new_df["UNIT PRICE"] != "")]

    if new_df.empty:
        print("❌ No valid data to append.")
        return

    # Convert "QUANTITY" column to integer
    if "Quantity" in new_df.columns:
        new_df["Quantity"] = pd.to_numeric(new_df["Quantity"], errors="coerce").fillna(0).astype(int)

    # Append new data to session state DataFrame
    session_state["df_store"] = pd.concat([session_state["df_store"], new_df], ignore_index=True)
