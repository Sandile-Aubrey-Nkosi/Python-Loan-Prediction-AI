
import tkinter as tk
from tkinter import Text, Scrollbar, Entry, Button, Label, END, NORMAL, DISABLED
import joblib
import pandas as pd
import pyttsx3  # Import the text-to-speech library

# Load your trained model
model = joblib.load('loan_status_predict.pkl')

BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

class ChatApplication:
    def __init__(self):
        self.window = tk.Tk()
        self._setup_main_window()
        self.engine = pyttsx3.init()  # Initialize the TTS engine
        self._set_female_voice()  # Set a female voice
        self.engine.setProperty('rate', 150)  # Set default speech rate

        # Variables to store user inputs for loan prediction
        self.reset_inputs()
        self.option_selected = None

        # Show welcome message
        self._insert_message("Welcome! Type 1 to check loan status, 0 to exit, or 'help' for assistance.", "Snipes")

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("Loan Status Chatbot")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=470, height=550, bg=BG_COLOR)

        # Head label
        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR,
                           text="Welcome to PyMLTeam-Loan Status Checker", font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)

        # Tiny divider
        line = Label(self.window, width=450, bg=BG_GRAY)
        line.place(relwidth=1, rely=0.07, relheight=0.012)

        # Text widget
        self.text_widget = Text(self.window, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR,
                                font=FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        # Scroll bar
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)

        # Bottom label
        bottom_label = Label(self.window, bg=BG_GRAY, height=80)
        bottom_label.place(relwidth=1, rely=0.825)

        # Message entry box
        self.msg_entry = Entry(bottom_label, bg="#2C3E50", fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.54, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)

        # Send button
        send_button = Button(bottom_label, text="Send", font=FONT_BOLD, width=20, bg=BG_GRAY,
                             command=lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.55, rely=0.008, relheight=0.06, relwidth=0.22)

        # Reset button
        reset_button = Button(bottom_label, text="Reset", font=FONT_BOLD, width=20, bg=BG_GRAY,
                              command=self.reset_chat)
        reset_button.place(relx=0.78, rely=0.008, relheight=0.06, relwidth=0.22)

    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg, "You")

    def _insert_message(self, msg, sender):
        if not msg:
            return

        self.msg_entry.delete(0, END)
        msg1 = f"{sender}: {msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)

        response = self.process_input(msg)
        msg2 = f"Snipes: {response}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state=DISABLED)

        # Speak the response
        self.speak(response)

        self.text_widget.see(END)

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def _set_female_voice(self):
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if 'female' in voice.name.lower():  # Check if the voice is female
                self.engine.setProperty('voice', voice.id)
                break

    def process_input(self, user_input):
        user_input = user_input.strip().lower()

        # Exit option
        if self.option_selected is None:
            if user_input == "0":
                self.window.quit()
                return "Goodbye! Have a great day!"
            elif user_input == "1":
                self.option_selected = 1  # Set option to proceed with loan check
                return "Great! Let's start with your Gender (1 for Male, 0 for Female)."
            elif user_input == "help":
                return ("Type 1 to check loan status, 0 to exit, or follow the prompts after selecting 1.")
            else:
                return "Please type 1 to check loan status, 0 to exit, or 'help' for assistance."

        # Process loan status steps after option is selected
        if self.gender is None:
            return self.get_gender(user_input)

        if self.married is None:
            return self.get_married(user_input)

        if self.dependents is None:
            return self.get_dependents(user_input)

        if self.education is None:
            return self.get_education(user_input)

        if self.self_employed is None:
            return self.get_self_employed(user_input)

        if self.applicant_income is None:
            return self.get_applicant_income(user_input)

        if self.coapplicant_income is None:
            return self.get_coapplicant_income(user_input)

        if self.loan_amount is None:
            return self.get_loan_amount(user_input)

        if self.loan_term is None:
            return self.get_loan_term(user_input)

        if self.credit_history is None:
            return self.get_credit_history(user_input)

        if self.property_area is None:
            return self.get_property_area(user_input)

    def get_gender(self, user_input):
        try:
            self.gender = int(user_input)
            if self.gender not in [0, 1]:
                raise ValueError
            return "Are you Married? (1 for Yes, 0 for No)."
        except ValueError:
            return "Invalid input. Please enter 1 for Male or 0 for Female."

    def get_married(self, user_input):
        try:
            self.married = int(user_input)
            if self.married not in [0, 1]:
                raise ValueError
            return "Please enter number of Dependents."
        except ValueError:
            return "Invalid input. Please enter 1 for Yes or 0 for No."

    def get_dependents(self, user_input):
        try:
            self.dependents = int(user_input)
            return "Are you Graduate? (1 for Yes, 0 for No)."
        except ValueError:
            return "Invalid input. Please enter a valid number of Dependents."

    def get_education(self, user_input):
        try:
            self.education = int(user_input)
            if self.education not in [0, 1]:
                raise ValueError
            return "Are you Self-Employed? (1 for Yes, 0 for No)."
        except ValueError:
            return "Invalid input. Please enter 1 for Graduate or 0 for Not Graduate."

    def get_self_employed(self, user_input):
        try:
            self.self_employed = int(user_input)
            if self.self_employed not in [0, 1]:
                raise ValueError
            return "Please enter Applicant Income."
        except ValueError:
            return "Invalid input. Please enter 1 for Yes or 0 for No."

    def get_applicant_income(self, user_input):
        try:
            self.applicant_income = float(user_input)
            return "Please enter Co-applicant Income."
        except ValueError:
            return "Invalid input. Please enter a valid income amount."

    def get_coapplicant_income(self, user_input):
        try:
            self.coapplicant_income = float(user_input)
            return "Please enter Loan Amount."
        except ValueError:
            return "Invalid input. Please enter a valid income amount."

    def get_loan_amount(self, user_input):
        try:
            self.loan_amount = float(user_input)
            return "Please enter Loan Term in months."
        except ValueError:
            return "Invalid input. Please enter a valid loan amount."

    def get_loan_term(self, user_input):
        try:
            self.loan_term = int(user_input)
            return "Please enter Credit History (1 for Good or 0 for Bad)."
        except ValueError:
            return "Invalid input. Please enter a valid loan term in months."

    def get_credit_history(self, user_input):
        try:
            self.credit_history = int(user_input)
            if self.credit_history not in [0, 1]:
                raise ValueError
            return "Please enter Property Area (1 for Rural, 2 for Semiurban, 3 for Urban)."
        except ValueError:
            return "Invalid input. Please enter 1 for Yes or 0 for No."

    def get_property_area(self, user_input):
        try:
            self.property_area = int(user_input)
            if self.property_area not in [1, 2, 3]:
                raise ValueError
            return self.make_prediction()
        except ValueError:
            return "Invalid input. Please enter 1 for Rural, 2 for Semiurban, or 3 for Urban."

    def make_prediction(self):
        # Prepare data for prediction
        input_data = pd.DataFrame({
            'Gender': [self.gender],
            'Married': [self.married],
            'Dependents': [self.dependents],
            'Education': [self.education],
            'Self_Employed': [self.self_employed],
            'ApplicantIncome': [self.applicant_income],
            'CoapplicantIncome': [self.coapplicant_income],
            'LoanAmount': [self.loan_amount],
            'Loan_Amount_Term': [self.loan_term],
            'Credit_History': [self.credit_history],
            'Property_Area': [self.property_area]
        })

        # Make prediction
        prediction = model.predict(input_data)
        self.reset_inputs()  # Reset inputs for next interaction

        if prediction[0] == 1:
            return "Congratulations! Your loan is approved!"
        else:
            return "Sorry, your loan is not approved. Please check your details and try again."

    def reset_chat(self):
        self.reset_inputs()
        self.text_widget.configure(state=NORMAL)
        self.text_widget.delete(1.0, END)
        self.text_widget.configure(state=DISABLED)
        self._insert_message("Chat reset. Type 1 to check loan status, 0 to exit, or 'help' for assistance.", "Snipes")

    def reset_inputs(self):
        self.gender = None
        self.married = None
        self.dependents = None
        self.education = None
        self.self_employed = None
        self.applicant_income = None
        self.coapplicant_income = None
        self.loan_amount = None
        self.loan_term = None
        self.credit_history = None
        self.property_area = None
        self.option_selected = None

if __name__ == "__main__":
    app = ChatApplication()
    app.run()
