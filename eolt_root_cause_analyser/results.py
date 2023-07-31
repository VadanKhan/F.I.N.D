import shutil

from fpdf import FPDF

local_path = "eolt_root_cause_analyser/results.pdf"


def results(status_input: dict, eol_test_id, test_type, test_id, remote_path):
    pdf = FPDF()

    # Add a page to the PDF
    pdf.add_page()

    # Set the font for the PDF
    pdf.set_font("Arial", size=20)
    init_string = f"Reporting on Test Data with eol test id: {eol_test_id}, test type: {test_type}, test id: {test_id}"
    pdf.multi_cell(0, 10, txt=init_string, align="L")
    pdf.set_font("Arial", size=12)
    for key in status_input.keys():
        # Add the lists to the PDF
        # pdf.cell(200, 10, txt=str(key), ln=0, align="L")
        # pdf.cell(200, 10, txt="blah", ln=1, align="L")
        pdf.cell(75, 7, str(key) + ":", ln=0, align="L")
        if status_input[key] == ["0", "0", "0", "0"]:
            pdf.cell(75, 7, "Passed.", ln=1, align="L")
        else:
            pdf.cell(75, 7, "Failed.", ln=1, align="L")

    # Save the PDF
    pdf.output("eolt_root_cause_analyser/results.pdf")
    if remote_path == 0:
        return 0
    else:
        shutil.copyfile(local_path, remote_path)
        return 0
