import streamlit as st
import pandas as pd
from io import BytesIO
import plotly.express as px
from transformers import pipeline
from datetime import datetime

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="Asytic Chatbot", page_icon="🧠", layout="centered")
st.title("🧠 Asytic Chatbot — Mental Health Companion")
st.caption("I’m here to listen and support. This is not a substitute for professional care.")

# ---------------------------
# Load models (primary + fallback)
# ---------------------------
@st.cache_resource
def load_models():
    try:
        primary = pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            return_all_scores=True
        )
        return primary, "goemotions"
    except Exception:
        backup = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
        return backup, "hartmann"

emotion_model, model_kind = load_models()

# ---------------------------
# Session state
# ---------------------------
if "profile" not in st.session_state:
    st.session_state.profile = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# ---------------------------
# Profile setup (expanded)
# ---------------------------
with st.expander("Set up your profile"):
    st.markdown("Your info stays on this device during this session.")

    col1, col2 = st.columns(2)  
    with col1:  
        name = st.text_input("Full Name", value=st.session_state.profile.get("name", ""))  
        age = st.number_input("Age", min_value=10, max_value=100, value=st.session_state.profile.get("age", 21))  
        occupation = st.text_input("Occupation", value=st.session_state.profile.get("occupation", "Student"))  
        sleep_hours = st.number_input("Average Sleep (hours)", min_value=0.0, max_value=16.0, value=float(st.session_state.profile.get("sleep_hours", 7)))  
    with col2:  
        gender = st.selectbox("Gender", ["Female", "Male", "Other"], index=0)  
        lifestyle = st.selectbox("Lifestyle", ["Sedentary", "Moderately Active", "Active"], index=1)  
        exercise_days = st.slider("Exercise Days / week", 0, 7, value=int(st.session_state.profile.get("exercise_days", 3)))  
        stressors = st.text_area("Major Stressors (comma-separated)", value=st.session_state.profile.get("stressors", ""))  

    st.markdown("Health background (optional)")  
    medical_history = st.text_area("Medical History")  
    medications = st.text_area("Current Medications")  
    therapy_history = st.text_area("Therapy/Counselling History")  

    if st.button("Save Profile"):  
        st.session_state.profile = {  
            "name": name,  
            "age": age,  
            "gender": gender,  
            "occupation": occupation,  
            "lifestyle": lifestyle,  
            "sleep_hours": sleep_hours,  
            "exercise_days": exercise_days,  
            "stressors": stressors,  
            "medical_history": medical_history,  
            "medications": medications,  
            "therapy_history": therapy_history,  
        }  
        st.success("Profile saved.")

# ---------------------------
# Safety: crisis keywords
# ---------------------------
CRISIS_KEYWORDS = {
    "self_harm": [
        "kill myself", "suicide", "end my life", "self harm", "self-harm", "cut myself",
        "don't want to live", "die", "hurting myself"
    ]
}

def needs_urgent_help(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in CRISIS_KEYWORDS["self_harm"])

def crisis_message():
    st.error(
        "🚨 If you feel you might harm yourself or someone else, please seek immediate help.\n\n"
        "- India: Call Kiran Helpline 1800-599-0019 or iCall 9152987821.\n"
        "- Elsewhere: Contact your local emergency number or a trusted person nearby."
    )

# ---------------------------
# Emotion detection utilities
# ---------------------------
def detect_emotions(text: str):
    raw = emotion_model(text)[0]
    df = pd.DataFrame(raw)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    top_label = df.loc[0, "label"]
    top_score = float(df.loc[0, "score"])
    return top_label, top_score, df

def supportive_reply(emotion: str, score: float, profile: dict, text: str):
    name = profile.get("name", "Friend")
    sadness_like = {"sadness", "grief", "disappointment", "remorse"}
    anxiety_like = {"nervousness", "fear", "embarrassment", "confusion", "worry"}
    anger_like = {"anger", "annoyance", "disapproval", "disgust", "jealousy"}
    joy_like = {"joy", "amusement", "excitement", "admiration", "love", "gratitude", "relief", "pride", "optimism"}
    strong = score >= 0.70  

    if needs_urgent_help(text):  
        return (
            f"{name}, I’m deeply concerned about your safety. Your life matters, and you’re not alone. "
            "Please contact a trusted person nearby and reach out to a trained professional right now. "
            "You can also call a helpline — I can share numbers for your country."
        )  

    if emotion in sadness_like:  
        if strong:  
            return (
                f"{name}, I can sense this {emotion.lower()} is weighing heavily on you. "
                "I want to reassure you that these feelings are valid and deserve compassionate care. "
                "Consider speaking with a licensed therapist or counsellor — they can help create a plan to ease this burden. "
                "Meanwhile, focus on rest, hydration, and gentle movement if possible."
            )  
        else:  
            return (
                f"{name}, I hear that you’re feeling some {emotion.lower()}. "
                "It’s completely okay to acknowledge this. Even small steps — like taking a short walk, journaling, or calling a friend — can make a difference."
            )  

    if emotion in anxiety_like:  
        if strong:  
            return (
                f"{name}, anxiety can be overwhelming. Please remember you’re safe right now. "
                "Consider booking time with a mental health professional who can teach grounding techniques tailored for you. "
                "For now, try this: name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, and 1 you taste."
            )  
        else:  
            return (
                f"{name}, I notice some {emotion.lower()}. Writing your worries down and focusing on just one actionable step today might help ease your mind."
            )  

    if emotion in anger_like:  
        return (
            f"{name}, your {emotion.lower()} is valid. If it feels intense, step away from the situation briefly. "
            "Breathing deeply and reflecting before responding can help — and if it’s recurring, a counsellor might help unpack the triggers."
        )  

    if emotion in joy_like:  
        return (
            f"That’s wonderful, {name}! Please take a moment to enjoy this {emotion.lower()} fully. "
            "Savoring these moments can strengthen emotional resilience."
        )  

    return (
        f"I’m here with you, {name}. Thank you for sharing openly. "
        "Would you like ideas for coping strategies, journaling prompts, or professional resources?"
    )

# ---------------------------
# Input UI
# ---------------------------
st.subheader("How are you feeling?")
st.session_state.user_input = st.text_area(
    "Type here…",
    value=st.session_state.user_input,
    key="input_area"
)

col_send, col_quick1, col_quick2, col_quick3 = st.columns([2,1,1,1])
with col_send:
    send_clicked = st.button("Send")
with col_quick1:
    if st.button("Quick: I feel anxious"):
        st.session_state.user_input = "I feel anxious and my thoughts are racing."
with col_quick2:
    if st.button("Quick: I feel low"):
        st.session_state.user_input = "I feel very low and unmotivated today."
with col_quick3:
    if st.button("Quick: I feel angry"):
        st.session_state.user_input = "I am frustrated and angry about what happened."

# ---------------------------
# Handle input
# ---------------------------
if send_clicked:
    text = st.session_state.user_input.strip()
    if text:
        top_emotion, top_score, df_scores = detect_emotions(text)
        st.session_state.chat_history.append({
            "user": text,
            "emotion": top_emotion,
            "score": float(top_score),
            "all_emotions": df_scores.to_dict(orient="records"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        bot = supportive_reply(top_emotion, top_score, st.session_state.profile, text)
        st.session_state.chat_history.append({"bot": bot, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        st.session_state.user_input = ""

# ---------------------------
# Conversation + Graphs
# ---------------------------
st.subheader("Conversation")
timeline_rows = []
for msg in st.session_state.chat_history:
    if "user" in msg:
        st.markdown(
            f"<div style='background-color:#DCEBFA; color:black; padding:12px; border-radius:12px; margin-bottom:8px;'>"
            f"<b>You:</b> {msg['user']}<br>"
            f"<i>Detected Emotion:</i> {msg['emotion']} (score: {msg['score']:.2f})</div>",
            unsafe_allow_html=True
        )
        df = pd.DataFrame(msg["all_emotions"])  
        df["score"] = df["score"].astype(float)  
        df = df.sort_values("score", ascending=False)  

        fig = px.bar(  
            df.head(10), x="label", y="score", color="label",  
            text=df.head(10)["score"].map(lambda x: f"{x:.2f}")  
        )  
        fig.update_traces(textposition="outside")  
        fig.update_layout(  
            title="Current message — emotion confidence (top 10)",  
            yaxis=dict(range=[0,1], title="Confidence"),  
            xaxis_title="Emotion", showlegend=False, height=420  
        )  
        st.plotly_chart(fig, use_container_width=True)  
        timeline_rows.append({"turn": len(timeline_rows)+1, "emotion": msg["emotion"], "score": msg["score"]})  

    elif "bot" in msg:  
        st.markdown(  
            f"<div style='background-color:#E8FFF0; color:black; padding:12px; border-radius:12px; margin-bottom:8px;'>"  
            f"<b>Asytic:</b> {msg['bot']}</div>",  
            unsafe_allow_html=True  
        )

if timeline_rows:
    st.subheader("Emotion confidence over time")
    tdf = pd.DataFrame(timeline_rows)
    fig2 = px.line(tdf, x="turn", y="score", markers=True)
    fig2.update_layout(xaxis_title="Message #", yaxis=dict(range=[0,1], title="Top emotion confidence"))
    st.plotly_chart(fig2, use_container_width=True)

if st.session_state.chat_history:
    latest_user = next((m for m in reversed(st.session_state.chat_history) if "user" in m), None)
    if latest_user and needs_urgent_help(latest_user["user"]):
        crisis_message()

# ---------------------------
# CSV export (professional)
# ---------------------------
if st.session_state.chat_history:
    st.subheader("Download Chat History")
    export_rows = []
    for msg in st.session_state.chat_history:
        if "user" in msg:
            export_rows.append({
                "Timestamp": msg.get("timestamp", ""),
                "Role": "User",
                "Message": msg["user"],
                "Detected Emotion": msg["emotion"],
                "Confidence Score": f"{msg['score']:.2f}"
            })
        elif "bot" in msg:
            export_rows.append({
                "Timestamp": msg.get("timestamp", ""),
                "Role": "Asytic",
                "Message": msg["bot"],
                "Detected Emotion": "",
                "Confidence Score": ""
            })
    edf = pd.DataFrame(export_rows)
    buf = BytesIO()
    edf.to_csv(buf, index=False)
    st.download_button("📥 Download Chat History (CSV)", buf.getvalue(), file_name="asytic_chat_history.csv", mime="text/csv")