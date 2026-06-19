import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import customtkinter as ctk
from tkinter import filedialog
import time
import threading
import speech_recognition as sr

# Light Theme
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# ================== LOAD DATA ==================
data = pd.read_csv(
    "spam.csv",
    encoding="latin-1",
    on_bad_lines="skip"
)

if len(data.columns) == 1:
    data = data[data.columns[0]].str.split(',', expand=True)

data = data.iloc[:, :2]
data.columns = ['label', 'message']

data['label'] = data['label'].map({
    'ham': 0,
    'spam': 1
})

data = data.dropna()

spam = data[data['label'] == 1]
ham = data[data['label'] == 0].sample(len(spam), random_state=42)

data = pd.concat([spam, ham])
data = data.sample(frac=1, random_state=42)

# ================== ML ==================
X = data['message'].astype(str)
y = data['label']

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X, y)

feature_names = vectorizer.get_feature_names_out()

history = []

# ================== FUNCTIONS ==================
def run_detection(msg):
    progress.start()

    time.sleep(0.5)

    vect = vectorizer.transform([msg])

    prediction = int(model.predict(vect)[0])
    prob = model.predict_proba(vect)[0][prediction] * 100

    words = msg.lower().split()

    spam_words = [
        w for w in words
        if w in feature_names
    ][:5]

    if prediction == 1:
        result = f"🚫 Spam ({prob:.2f}%)"
        result_label.configure(
            text=result,
            text_color="#cc0000"
        )
    else:
        result = f"✅ Not Spam ({prob:.2f}%)"
        result_label.configure(
            text=result,
            text_color="#008a00"
        )

    if spam_words:
        highlight_label.configure(
            text="⚠ Keywords: " +
            ", ".join(spam_words)
        )
    else:
        highlight_label.configure(text="")

    history.append(result)

    history_box.insert(
        "end",
        f"{msg[:50]}... → {result}\n"
    )

    history_box.see("end")

    progress.stop()


def check_spam():
    msg = textbox.get(
        "1.0",
        "end"
    ).strip()

    if msg == "":
        result_label.configure(
            text="⚠ Enter message",
            text_color="orange"
        )
        return

    threading.Thread(
        target=run_detection,
        args=(msg,),
        daemon=True
    ).start()


def clear_text():
    textbox.delete("1.0", "end")
    result_label.configure(text="")
    highlight_label.configure(text="")


def delete_history():
    history.clear()
    history_box.delete("1.0", "end")


def upload_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Text Files", "*.txt")]
    )

    if file_path:
        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            textbox.delete("1.0", "end")
            textbox.insert(
                "end",
                f.read()
            )


# ================== VOICE INPUT ==================
def voice_input():

    recognizer = sr.Recognizer()

    result_label.configure(
        text="🎤 Listening...",
        text_color="blue"
    )

    app.update()

    try:
        with sr.Microphone() as source:

            recognizer.energy_threshold = 300
            recognizer.pause_threshold = 1.0

            recognizer.adjust_for_ambient_noise(
                source,
                duration=1
            )

            audio = recognizer.listen(
                source,
                timeout=5,
                phrase_time_limit=6
            )

        result_label.configure(
            text="Processing...",
            text_color="blue"
        )

        app.update()

        text = recognizer.recognize_google(audio)

        textbox.delete("1.0", "end")
        textbox.insert("end", text)

        result_label.configure(
            text="Voice captured ✔",
            text_color="green"
        )

    except sr.WaitTimeoutError:
        result_label.configure(
            text="⏱ No speech detected",
            text_color="orange"
        )

    except sr.UnknownValueError:
        result_label.configure(
            text="❌ Could not understand voice",
            text_color="red"
        )

    except Exception as e:
        print(e)

        result_label.configure(
            text="Mic error",
            text_color="red"
        )


# ================== GUI ==================
app = ctk.CTk()

app.title("AI Email Spam Detector")
app.geometry("720x800")

title = ctk.CTkLabel(
    app,
    text="🤖 AI Email Spam Detector",
    font=("Arial", 26, "bold")
)

title.pack(pady=15)

textbox = ctk.CTkTextbox(
    app,
    height=140
)

textbox.pack(
    padx=20,
    pady=10,
    fill="x"
)

btn_frame = ctk.CTkFrame(
    app,
    fg_color="transparent"
)

btn_frame.pack()

ctk.CTkButton(
    btn_frame,
    text="🔍 Check",
    command=check_spam
).grid(row=0, column=0, padx=8)

ctk.CTkButton(
    btn_frame,
    text="🧹 Clear",
    command=clear_text
).grid(row=0, column=1, padx=8)

ctk.CTkButton(
    btn_frame,
    text="📁 Upload",
    command=upload_file
).grid(row=0, column=2, padx=8)

ctk.CTkButton(
    btn_frame,
    text="🎤 Voice",
    command=voice_input
).grid(row=0, column=3, padx=8)

progress = ctk.CTkProgressBar(app)

progress.pack(
    padx=20,
    pady=8,
    fill="x"
)

progress.set(0)

result_label = ctk.CTkLabel(
    app,
    text="",
    font=("Arial", 20, "bold")
)

result_label.pack(pady=8)

highlight_label = ctk.CTkLabel(
    app,
    text="",
    font=("Arial", 14)
)

highlight_label.pack()

history_header = ctk.CTkFrame(
    app,
    fg_color="transparent"
)

history_header.pack(
    fill="x",
    pady=5
)

ctk.CTkLabel(
    history_header,
    text="📜 Detection History",
    font=("Arial", 18, "bold")
).pack(
    side="left",
    padx=20
)

ctk.CTkButton(
    history_header,
    text="🗑 Delete History",
    width=140,
    command=delete_history,
    fg_color="#cc3333"
).pack(
    side="right",
    padx=20
)

history_box = ctk.CTkTextbox(
    app,
    height=200
)

history_box.pack(
    padx=20,
    pady=10,
    fill="both",
    expand=True
)

app.mainloop()