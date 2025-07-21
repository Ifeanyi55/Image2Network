from CV2Net import cv2net
import gradio as gr
import pandas as pd

def image2net(img_file,api_key):
  try:
    if isinstance(img_file, list) and len(img_file) > 1:
      df_list = []
      for i in img_file:
        df_list.append(cv2net(i,api_key))

    else:
      df_list = [cv2net(img_file,api_key)]

  except Exception:
      gr.Info("The model is overloaded. Please try again later!")

  # Filter out None values before concatenating
  valid_dfs = [df for df in df_list if df is not None]

  if valid_dfs:
    df = pd.concat(valid_dfs)
    file_path = "network_data.csv"
    df.to_csv(file_path, index=False)
    return df, file_path
  else:
    return None, None
