import io

import PIL.Image
import PySimpleGUI as sg


def show_results(company, field, slogan_model, slogan_gpt, banner_model_path, banner_gpt_path, keywords):
    sg.theme("DarkTeal12")

    layout = [
        [sg.Text(f"Company name: {company}    Field: {field}      Optimization search keywords: {keywords}", background_color="white",
                 expand_x=True, expand_y=True, text_color="blue")],
        [sg.Text("Custom model slogan: ")],
        [sg.Output(key='cust_slogan', size=(60, 1), expand_x=True, expand_y=True)],
        [sg.HSeparator()],
        [sg.Text("OpenAI model slogan: ")],
        [sg.Output(key='open_slogan', size=(60, 1), expand_x=True, expand_y=True)],
        [sg.HSeparator()],
        [sg.Image(key="openai_banner"), sg.VSeparator(), sg.Image(key="model_banner")]]


    window = sg.Window('Generated advertisements', layout, finalize=True, resizable=True)
    try:
        window['cust_slogan'].print(slogan_model, text_color='blue')
        window['open_slogan'].print(slogan_gpt, text_color='blue')
        model_image = PIL.Image.open(banner_model_path).resize((300, 300))
        openai_image = PIL.Image.open(banner_gpt_path).resize((300, 300))


        model_bytes = io.BytesIO()
        model_image.save(model_bytes, format='PNG')
        model_data = model_bytes.getvalue()
        window["model_banner"].update(data=model_data)

        openai_bytes = io.BytesIO()
        openai_image.save(openai_bytes, format='PNG')
        openai_data = openai_bytes.getvalue()
        window["openai_banner"].update(data=openai_data)


    except Exception as e:
        sg.popup("Error: ", e)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break

