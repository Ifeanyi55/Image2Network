from ImageNet import image2net
from CV2Net import cv2net
from Download import save_csv
import gradio as gr

css = css = """
  .action-btn {
    align-items: center;
    appearance: none;
    background-image: radial-gradient(100% 100% at 100% 0, #5aff76 0, #07911a 100%);
    border: 0;
    border-radius: 6px;
    box-shadow: rgba(45, 35, 66, .4) 0 2px 4px,rgba(45, 35, 66, .3) 0 7px 13px -3px,rgba(58, 65, 111, .5) 0 -3px 0 inset;
    box-sizing: border-box;
    color: #0c0b0b;
    cursor: pointer;
    display: inline-flex;
    font-family: "JetBrains Mono",monospace;
    height: 48px;
    justify-content: center;
    line-height: 1;
    list-style: none;
    overflow: hidden;
    padding-left: 16px;
    padding-right: 16px;
    position: relative;
    text-align: left;
    text-decoration: none;
    transition: box-shadow .15s,transform .15s;
    user-select: none;
    -webkit-user-select: none;
    touch-action: manipulation;
    white-space: nowrap;
    will-change: box-shadow,transform;
    font-size: 18px;
  }

  .action-btn:focus {
    outline: none;
    box-shadow: none
  }

  .action-btn:hover {
    box-shadow: rgba(45, 35, 66, .4) 0 4px 8px, rgba(45, 35, 66, .3) 0 7px 13px -3px, #3ce065 0 -3px 0 inset;
    transform: translateY(-2px);
  }

  .action-btn:active {
    box-shadow: #3ce065 0 3px 7px inset;
    transform: translateY(2px);
  }

  /* Enhanced dark theme for dataframe - Multiple selectors for compatibility */
  #dark-df,
  #dark-df .MuiDataGrid-root,
  #dark-df .dataframe,
  #dark-df .gr-dataframe,
  #dark-df table {
    background-color: #0a0a0a !important;
    color: #e8e8e8 !important;
    border: 2px solid #404040 !important;
    border-radius: 8px !important;
    font-size: 14px;
  }

  /* Headers for both MUI and standard tables */
  #dark-df .MuiDataGrid-columnHeaders,
  #dark-df thead,
  #dark-df th {
    background-color: #1a1a1a !important;
    color: #ffffff !important;
    border-bottom: 2px solid #505050 !important;
    font-weight: 600 !important;
  }

  #dark-df .MuiDataGrid-columnHeader {
    border-right: 1px solid #404040 !important;
  }

  /* Cells for both MUI and standard tables */
  #dark-df .MuiDataGrid-cell,
  #dark-df td {
    background-color: #0a0a0a !important;
    color: #e8e8e8 !important;
    border-bottom: 1px solid #333333 !important;
    border-right: 1px solid #333333 !important;
    padding: 8px 12px !important;
  }

  #dark-df .MuiDataGrid-cell:last-of-type {
    border-right: none !important;
  }

  /* Rows for both MUI and standard tables */
  #dark-df .MuiDataGrid-row,
  #dark-df tr {
    background-color: #0a0a0a !important;
  }

  #dark-df .MuiDataGrid-row:nth-of-type(even),
  #dark-df tr:nth-of-type(even) {
    background-color: #111111 !important;
  }

  #dark-df .MuiDataGrid-row:hover,
  #dark-df tr:hover {
    background-color: #1f1f1f !important;
  }

  #dark-df .MuiDataGrid-row:nth-of-type(even):hover,
  #dark-df tr:nth-of-type(even):hover {
    background-color: #1f1f1f !important;
  }

  #dark-df .MuiDataGrid-footerContainer {
    background-color: #1a1a1a !important;
    color: #cccccc !important;
    border-top: 2px solid #505050 !important;
  }

  /* Add Gradio-specific dataframe styling */
  #dark-df .gr-dataframe,
  #dark-df .gr-dataframe table,
  #dark-df .gr-dataframe tbody,
  #dark-df .gr-dataframe thead {
    background-color: #0a0a0a !important;
    color: #e8e8e8 !important;
  }

  #dark-df .gr-dataframe th {
    background-color: #1a1a1a !important;
    color: #ffffff !important;
    border: 1px solid #404040 !important;
  }

  #dark-df .gr-dataframe td {
    background-color: #0a0a0a !important;
    color: #e8e8e8 !important;
    border: 1px solid #333333 !important;
  }

  #dark-df .gr-dataframe tr:nth-child(even) td {
    background-color: #111111 !important;
  }

  #dark-df .gr-dataframe tr:hover td {
    background-color: #1f1f1f !important;
  }

  /* Force override any default styling */
  #dark-df * {
    box-sizing: border-box;
  }

  #dark-df .gr-dataframe .gr-button {
    background-color: #333333 !important;
    color: #e8e8e8 !important;
    border: 1px solid #555555 !important;
  }

  #dark-df .gr-dataframe .gr-button:hover {
    background-color: #444444 !important;
  }
  #dark-df .MuiDataGrid-virtualScroller::-webkit-scrollbar {
    width: 8px;
    height: 8px;
  }

  #dark-df .MuiDataGrid-virtualScroller::-webkit-scrollbar-track {
    background: #1a1a1a;
  }

  #dark-df .MuiDataGrid-virtualScroller::-webkit-scrollbar-thumb {
    background: #404040;
    border-radius: 4px;
  }

  #dark-df .MuiDataGrid-virtualScroller::-webkit-scrollbar-thumb:hover {
    background: #505050;
  }

  /* Selection styling */
  #dark-df .MuiDataGrid-row.Mui-selected {
    background-color: #2d4a3e !important;
  }

  #dark-df .MuiDataGrid-row.Mui-selected:hover {
    background-color: #365242 !important;
  }

  #dark-df .MuiDataGrid-cell.Mui-selected {
    background-color: #2d4a3e !important;
  }

  /* Sorting and filtering icons */
  #dark-df .MuiDataGrid-iconSeparator {
    color: #666666 !important;
  }

  #dark-df .MuiDataGrid-sortIcon {
    color: #cccccc !important;
  }

  #dark-df .MuiDataGrid-filterIcon {
    color: #cccccc !important;
  }
"""
with gr.Blocks(
    theme=gr.themes.Base(primary_hue="teal", secondary_hue="amber").set(
    body_background_fill="*neutral_950",
    body_text_color="*neutral_200",
    background_fill_primary="*neutral_900",
    background_fill_secondary="*neutral_800",
    border_color_primary="*neutral_700",
    block_background_fill="*neutral_900",
    input_background_fill="*neutral_800",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_secondary_background_fill="*neutral_700",
    button_secondary_background_fill_hover="*neutral_600"
),
    css=css,
) as app:
  gr.HTML("<h1 style='text-align:center;color:#07911a;font-weight:bold;'>🖼️ Image To Network 🕸️</h1>")
  with gr.Column(elem_classes="cols"):
    with gr.Row():
      gr.Markdown("""
      How To Use:
      - Visit [Google AI Studio](https://aistudio.google.com/app/apikey) and obtain your Google API key.
      - Paste your Google API key
      - Upload image files from local folder
      - Generate network data from uploaded images
      - Download data for analysis in NodeXL
      """)
    with gr.Row():
      api_key = gr.Textbox(label="🔑 Google API Key",placeholder="Paste your Google API Key",elem_classes="widgets")
    with gr.Row():
      image_input = gr.File(file_types=["image"],
                      label="🖼️ Upload Image Files",
                      file_count="multiple",
                      elem_classes="widgets",
                      interactive=True)
      df_output = gr.Dataframe(label="🕸️ Network Data", elem_id="dark-df")
    with gr.Column(elem_classes="cols"):
      with gr.Row():
        btn = gr.Button("🧬 Generate Data",elem_classes="action-btn", elem_id = "click")
        download_btn = gr.DownloadButton("⬇️ Download Data", elem_classes="action-btn")
      with gr.Column():
        with gr.Row():
          clear = gr.Button("🗑️ Clear",value="", elem_classes="action-btn")


  btn.click(image2net, inputs=[image_input,api_key], outputs=[df_output, download_btn])
  clear.click(lambda: [None,None,None], inputs = [], outputs = [image_input, df_output])

 
if __name__ == "__main__":
  app.launch()