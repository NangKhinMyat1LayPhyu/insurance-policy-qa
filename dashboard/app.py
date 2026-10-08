import os
import sqlite3
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.db import DB_FILE, init_db, set_resolved

SERIES_COLOR = {"light": "#2a78d6", "dark": "#3987e5"}

st.set_page_config(page_title="Kasturi Support Admin", page_icon="🛡️", layout="wide")


def require_password():
    password = os.getenv("ADMIN_PASSWORD")
    if not password or st.session_state.get("authed"):
        return
    entered = st.text_input("Admin password", type="password")
    if entered == password:
        st.session_state.authed = True
        st.rerun()
    elif entered:
        st.error("Wrong password.")
    st.stop()


def load_messages():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM messages ORDER BY time", conn, parse_dates=["time"])
    conn.close()
    df["category"] = df["category"].fillna("Unclassified")
    return df


def series_color():
    theme = getattr(getattr(st.context, "theme", None), "type", None)
    return SERIES_COLOR.get(theme, SERIES_COLOR["light"])


def bar_chart(data, x, y, x_title, y_title, horizontal=False):
    color = series_color()
    bars = alt.Chart(data).mark_bar(color=color, cornerRadiusEnd=4)
    if horizontal:
        enc = dict(
            x=alt.X(f"{x}:Q", title=x_title, axis=alt.Axis(tickMinStep=1, grid=True)),
            y=alt.Y(f"{y}:N", title=None, sort="-x", axis=alt.Axis(labelLimit=260)),
        )
    else:
        enc = dict(
            x=alt.X(f"{x}:N", title=x_title, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{y}:Q", title=y_title, axis=alt.Axis(tickMinStep=1, grid=True)),
        )
    tooltip = [alt.Tooltip(f"{y if horizontal else x}:N", title=y_title if horizontal else x_title),
               alt.Tooltip(f"{x if horizontal else y}:Q", title="Questions")]
    chart = bars.encode(**enc, tooltip=tooltip).configure_axis(domain=False, tickColor="transparent", gridOpacity=0.4)
    st.altair_chart(chart, width="stretch")


def needs_human_tab(df):
    show_resolved = st.toggle("Show resolved", value=False)
    flagged = df[df["needs_human"] == 1]
    if not show_resolved:
        flagged = flagged[flagged["resolved"] == 0]

    if flagged.empty:
        st.success("✅ Nothing waiting. The bot handled every question.")
        return

    for row in flagged.sort_values("time", ascending=False).itertuples():
        with st.container(border=True):
            left, right = st.columns([5, 1])
            status = "✅ Resolved" if row.resolved else "🚩 Open"
            left.markdown(f"**{status} · {row.flag_reason}**  \n"
                          f"<small>{row.time:%d %b %Y, %H:%M} · conversation `{row.session_id[:8]}`</small>",
                          unsafe_allow_html=True)
            if row.resolved:
                if right.button("Reopen", key=f"reopen-{row.id}"):
                    set_resolved(row.id, False)
                    st.rerun()
            elif right.button("Mark resolved", key=f"resolve-{row.id}", type="primary"):
                set_resolved(row.id, True)
                st.rerun()
            st.markdown(f"> {row.question}")
            with st.expander("What the bot replied"):
                st.write(row.answer)


def conversations_tab(df):
    show = st.radio("Show", ["All", "Flagged only", "Cache hits only"], horizontal=True, label_visibility="collapsed")
    search = st.text_input("Search questions", placeholder="e.g. flood, waiting period")

    view = df
    if show == "Flagged only":
        view = view[view["needs_human"] == 1]
    elif show == "Cache hits only":
        view = view[view["cache_hit"] == 1]
    if search:
        view = view[view["question"].str.contains(search, case=False, regex=False)]
    if view.empty:
        st.info("No conversations match.")
        return

    sessions = (
        view.groupby("session_id")
        .agg(started=("time", "min"), last_message=("time", "max"), messages=("id", "count"),
             flagged=("needs_human", "max"), first_question=("question", "first"))
        .sort_values("last_message", ascending=False)
        .reset_index()
    )
    sessions["flagged"] = sessions["flagged"].map({1: "🚩 Yes", 0: ""})
    sessions["conversation"] = sessions["session_id"].str[:8]

    picked = st.dataframe(
        sessions[["conversation", "started", "messages", "flagged", "first_question"]],
        hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
        column_config={
            "started": st.column_config.DatetimeColumn("Started", format="D MMM, HH:mm"),
            "first_question": st.column_config.TextColumn("First question", width="large"),
        },
    )
    rows = picked.selection.rows
    if not rows:
        st.caption("Select a conversation to read the full transcript.")
        return

    session_id = sessions.iloc[rows[0]]["session_id"]
    st.subheader(f"Conversation {session_id[:8]}")
    for msg in df[df["session_id"] == session_id].itertuples():
        with st.chat_message("user"):
            st.write(msg.question)
            if pd.notna(msg.standalone_question):
                st.caption(f"Understood as: {msg.standalone_question}")
        with st.chat_message("assistant"):
            st.write(msg.answer)
            notes = [f"{msg.time:%H:%M}", msg.category]
            if msg.cache_hit:
                notes.append(f"cache hit ({msg.similarity:.3f})")
            if msg.needs_human:
                notes.append(f"🚩 {msg.flag_reason}")
            st.caption(" · ".join(notes))


def insights_tab(df):
    left, right = st.columns(2)
    with left:
        st.markdown("**Questions per day**")
        per_day = df.groupby(df["time"].dt.strftime("%d %b")).size().reset_index(name="questions")
        per_day.columns = ["day", "questions"]
        bar_chart(per_day, "day", "questions", "Day", "Questions")
    with right:
        st.markdown("**What customers ask about**")
        topics = df.groupby("category").size().reset_index(name="questions").nlargest(10, "questions")
        bar_chart(topics, "questions", "category", "Questions", "Topic", horizontal=True)

    st.markdown("**How questions were handled**")
    outcome = df["flag_reason"].fillna("Answered by the bot")
    handled = outcome.value_counts().rename_axis("Outcome").reset_index(name="Questions")
    handled["Share"] = (handled["Questions"] / len(df)).map("{:.0%}".format)
    st.dataframe(handled, hide_index=True, width="stretch")


require_password()
init_db()
df = load_messages()

st.title("Kasturi support · admin")
st.caption("Every customer question the bot handled, what it couldn't answer, and what people ask about.")

if df.empty:
    st.info("No conversations yet. Ask the bot something at http://localhost:8000, "
            "or run `python3 scripts/seed_demo.py` to send a set of sample questions.")
    st.stop()

open_flags = int(((df["needs_human"] == 1) & (df["resolved"] == 0)).sum())
answered = 1 - df["needs_human"].mean()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Questions", len(df), border=True)
k2.metric("Conversations", df["session_id"].nunique(), border=True)
k3.metric("Answered by the bot", f"{answered:.0%}", border=True)
k4.metric("Waiting for a human", open_flags, border=True)

needs_tab, convo_tab, insight_tab = st.tabs([f"🚩 Needs a human ({open_flags})", "💬 Conversations", "📊 Insights"])
with needs_tab:
    needs_human_tab(df)
with convo_tab:
    conversations_tab(df)
with insight_tab:
    insights_tab(df)
