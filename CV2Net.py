# define analysis engine
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from google.genai import types
from google import genai
from io import BytesIO
from PIL import Image
import pandas as pd
import gradio as gr
import base64
import json
import os


def cv2net(image_path,api_key):
  # authenticate gemini client
  client = genai.Client(api_key=api_key)

  # call Google Search tool
  google_search_tool = Tool(
      google_search = GoogleSearch()
  )

  with open(image_path, 'rb') as f:
    image_data = f.read()

  prompt = """
  I want you to carefully analyze the image(s) and map the functional relationship between every single identified entity in the image.
  Do not ignore small or partially visible items. Collect the following information from the image(s) and DO NOT include items, objects, or things that are not in the image(s):
  - Specific object name or person
  - Precise functional relationship verb
  - Class: object, person, animal, environment, text, brand
  - Primary function or role
  - Dominant color
  - Small, medium, large, tiny, huge
  - Material type
  - Location description
  - Current condition
  - Spatial context
  - Setting or environment
  - Relationship strength: strong, medium, weak
  - Spatial context
  - Scene context
  - Confidence: high, medium, low
  - Today's date (YYYY-MM-DD)
  Ignore what a person in an image is wearing. Return the results as one JSON file with the following structure exactly:
  ```json
    [
      {
        "Vertex1": "specific_object_name_or_person",
        "Vertex2": "specific_object_name_or_person",
        "Relationship": "precise_functional_relationship_verb",
        "Vertex1_class": "Object|Person|Animal|Environment|Text|Brand",
        "Vertex1_purpose": "primary_function_or_role",
        "Vertex1_size": "small|medium|large|tiny|huge",
        "Vertex1_position": "location_description",
        "Vertex1_state": "current_condition",
        "Vertex2_class": "Object|Person|Animal|Environment|Text|Brand",
        "Vertex2_purpose": "primary_function_or_role",
        "Vertex2_size": "small|medium|large|tiny|huge",
        "Vertex2_position": "location_description",
        "Vertex2_state": "current_condition",
        "Relationship_type": "spatial|functional|contextual|interactive",
        "Relationship_strength": "strong|medium|weak",
        "Spatial_context": "detailed_spatial_description",
        "Scene_context": "setting_or_environment",
        "Confidence": "high|medium|low",
        "Date": "today's_date"
      }
    ]
    ```
Here is an example JSON output:
```json
    [
      {
        "Vertex1": "Man",
        "Vertex2": "Bench",
        "Relationship": "Sits on",
        "Vertex1_class": "Person",
        "Vertex1_purpose": "Posing for photo",
        "Vertex1_size": "Medium",
        "Vertex1_position": "Left foreground",
        "Vertex1_state": "Visible",
        "Vertex2_class": "Object",
        "Vertex2_purpose": "A seat",
        "Vertex2_size": "Medium",
        "Vertex2_position": "Middle ground",
        "Vertex2_state": "Visible",
        "Relationship_type": "Functional",
        "Relationship_strength": "Strong",
        "Spatial_context": "Man is sitting on bench",
        "Scene_context": "Outdoor scene in the park",
        "Confidence": "High",
        "Date": "2025-07-16"
      }
    ]
    ```
"""

  response = client.models.generate_content(
      model="gemini-2.0-flash",
      contents=[types.Part.from_bytes(data=image_data, mime_type="image/jpeg"), prompt],
      config=GenerateContentConfig(
          tools=[google_search_tool],
          response_modalities=["TEXT"],
          response_mime_type="application/json",
      )
  )

  try:
    # convert response from string to JSON
    json_file = json.loads(response.text)

    # convert JSON into a DataFrame
    df = pd.DataFrame(json_file)
    return df
  except json.JSONDecodeError as e:
    print(f"Error decoding JSON for image: {image_data} - {e}")
    return None
