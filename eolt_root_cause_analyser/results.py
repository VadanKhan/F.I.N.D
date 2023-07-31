from fpdf import FPDF


def results(status_input: dict):
    pdf = FPDF()

    # Add a page to the PDF
    pdf.add_page()

    # Set the font for the PDF
    pdf.set_font("Comic", size=12)
    for key in status_input.keys():
        # Add the lists to the PDF
        pdf.cell(200, 10, txt=status_input[key], ln=1, align="L")

    # Save the PDF
    pdf.output("eolt_root_cause_analyser/results.pdf")
