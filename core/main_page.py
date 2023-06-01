import PySimpleGUI as sg

from core import result_page
from core.SEO_script import retrieve_keywords
from core.inference import generate_slogans, generate_banner


def main():
    sg.theme("DarkTeal12")

    layout = [
        [sg.Text('Company name'), sg.InputText(key='company')],
        [sg.Text('Field'), sg.InputText(key='field')],
        [sg.Text('Use SEO keywords?')],
        [sg.Radio('Yes', 'radio_group', key='yes'), sg.Radio('No', 'radio_group', key='no')],
        [sg.Button('Generate ADV'),
        [sg.ProgressBar(5, orientation='h', size=(35, 20), border_width=4, key="-PROGRESS_BAR-")]]
    ]

    window = sg.Window('Advertisement generator', layout)

    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            break
        elif event == 'Generate ADV':
            if values['company'] == '' or values['field'] == '':
                sg.popup_error('Error', 'Please, specify a value for both fields')
            else:
                i = 0
                if values['yes']:
                    seo_optimization = retrieve_keywords(values['company'])
                    window["-PROGRESS_BAR-"].update(i + 1)
                    i += 1
                else:
                    seo_optimization = " "

                slogan_model, slogan_gpt = generate_slogans(values['company'], values['field'], seo_optimization)
                window["-PROGRESS_BAR-"].update(i + 1)
                i += 1

                banner_model_path, banner_gpt_path = generate_banner(values['company'], values['field'])
                window["-PROGRESS_BAR-"].update(i + 1)
                i += 1

                result_page.show_results(values['company'], values['field'], slogan_model, slogan_gpt,
                                         banner_model_path, banner_gpt_path, seo_optimization)
    window.close()


if __name__ == '__main__':
    main()
