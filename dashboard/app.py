import os
import sqlite3
import sys
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import library
from ingest import DOC_LABELS, DOCUMENTS
from server.db import DB_FILE, init_db, set_resolved

DEFAULT_TEST_QUESTIONS = """How long is the waiting period for a Specified Illness?
Water leaked from a pipe for 10 days before I noticed it. I'm Silver tier. Is the damage covered?
What fee is deducted from a Section B cancellation refund?"""
MAX_TEST_QUESTIONS = 4
TEST_PAUSE_SECONDS = 10

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


def flash(message):
    st.session_state.flash = message


def documents_tab():
    if library.library_dir() is None:
        st.info("Document management runs in Docker, where the document library is set up (LIBRARY_DIR).")
        return
    library.ensure_initialized()
    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))

    live_id = library.active_release_id()
    live = library.read_manifest(library.release_dir(live_id))
    st.markdown("**Live documents**")
    st.dataframe(
        [{"Document": doc["label"], "File": live["documents"][doc["doc_type"]]["file"],
          "Version": live["documents"][doc["doc_type"]]["version"],
          "Effective from": live["documents"][doc["doc_type"]]["effective_date"] or "—",
          "Pages": live["documents"][doc["doc_type"]]["pages"]} for doc in DOCUMENTS],
        hide_index=True, width="stretch",
    )
    st.caption(f"Live release `{live_id}`, published {live['published']}")

    staged_id = st.session_state.get("staged_release")
    if staged_id and not (library.release_dir(staged_id, staged=True) / "manifest.json").exists():
        st.session_state.pop("staged_release")
        staged_id = None
    if staged_id:
        review_staged(staged_id)
    else:
        replace_form()
    release_history(live_id)


def replace_form():
    st.markdown("**Replace a document**")
    with st.form("replace"):
        doc_type = st.selectbox("Document", [doc["doc_type"] for doc in DOCUMENTS], format_func=DOC_LABELS.get)
        upload = st.file_uploader("New PDF", type=["pdf"])
        left, right = st.columns(2)
        version = left.text_input("Version label", placeholder="e.g. v2 (2027-01)")
        effective = right.date_input("Effective from", value=None)
        note = st.text_input("Note (optional)", placeholder="What changed")
        submitted = st.form_submit_button("Check this PDF", type="primary")
    if not submitted:
        return
    if upload is None or not version.strip():
        st.error("Choose a PDF and give it a version label.")
        return
    with st.spinner("Reading the PDF and building its search index..."):
        try:
            staged_id = library.stage_replacement(doc_type, upload.getvalue(), upload.name, version.strip(),
                                                  effective.isoformat() if effective else None, note.strip())
        except library.LibraryError as e:
            st.error(str(e))
            return
    st.session_state.staged_release = staged_id
    st.rerun()


def review_staged(staged_id):
    reports = st.session_state.setdefault("reports", {})
    if staged_id not in reports:
        with st.spinner("Comparing with the live version..."):
            reports[staged_id] = library.check_report(staged_id)
    report = reports[staged_id]
    old, new = report["old"], report["new"]

    st.markdown(f"**Review before publishing: {report['label']}** · {old['version']} → {new['version']}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pages", new["pages"], new["pages"] - old["pages"], delta_color="off", border=True)
    m2.metric("Passages", new["passages"], new["passages"] - old["passages"], delta_color="off", border=True)
    m3.metric("Table rows", new["table_rows"], new["table_rows"] - old["table_rows"], delta_color="off", border=True)
    m4.metric("Sections found", ", ".join(new["sections"]) or "none", border=True)

    for message in report["errors"]:
        st.error(message, icon="⛔")
    for message in report["warnings"]:
        st.warning(message, icon="⚠️")
    if not report["errors"] and not report["warnings"]:
        st.success("No problems found when reading this PDF.", icon="✅")

    st.markdown(f"**What changed in the text**: {len(report['added'])} sentences added or reworded, "
                f"{len(report['removed'])} removed or reworded")
    for title, sentences, sign in (("New in this version", report["added"], "+"), ("No longer in this version", report["removed"], "−")):
        with st.expander(f"{title} ({len(sentences)})"):
            for sentence in sentences[:60]:
                st.markdown(f"{sign} {sentence}")
            if len(sentences) > 60:
                st.caption(f"...and {len(sentences) - 60} more")

    with st.expander("Test sample questions on the live and new versions"):
        questions_text = st.text_area("One question per line", DEFAULT_TEST_QUESTIONS, height=110)
        questions = [q.strip() for q in questions_text.splitlines() if q.strip()][:MAX_TEST_QUESTIONS]
        st.caption(f"Uses {len(questions) * 4} Gemini requests and takes about {len(questions) * 25} seconds.")
        if st.button("Run test"):
            run_comparison(staged_id, questions)
        for question, live_answer, new_answer in st.session_state.get("comparisons", {}).get(staged_id, []):
            st.markdown(f"**{question}**")
            left, right = st.columns(2)
            with left.container(border=True):
                st.caption("Live version")
                st.write(live_answer)
            with right.container(border=True):
                st.caption("New version")
                st.write(new_answer)

    confirmed = st.checkbox("I've reviewed the warnings above") if report["warnings"] else True
    left, right, _ = st.columns([1, 1, 3])
    if left.button("Publish", type="primary", disabled=bool(report["errors"]) or not confirmed):
        library.publish(staged_id)
        st.session_state.pop("staged_release", None)
        flash(f"Published {report['label']} {new['version']}. The bot is now answering from it.")
        st.rerun()
    if right.button("Discard"):
        library.discard_staged(staged_id)
        st.session_state.pop("staged_release", None)
        st.rerun()


def run_comparison(staged_id, questions):
    import qa
    from google.genai import errors as genai_errors

    staged_dir = library.release_dir(staged_id, staged=True)
    results = []
    progress = st.progress(0.0, text="Asking the live version...")
    try:
        for i, question in enumerate(questions):
            progress.progress(i / len(questions), text=f"Question {i + 1} of {len(questions)}")
            live_answer = qa.answer(question, use_cache=False)[0]
            time.sleep(TEST_PAUSE_SECONDS)
            new_answer = qa.answer(question, release=staged_dir)[0]
            time.sleep(TEST_PAUSE_SECONDS)
            results.append((question, live_answer, new_answer))
    except genai_errors.APIError:
        st.error("Gemini is busy (free-tier limit). Wait a minute and run the test again.")
    progress.empty()
    st.session_state.setdefault("comparisons", {})[staged_id] = results


def release_history(live_id):
    st.markdown("**History**")
    st.caption("Every published set of documents is kept. Making an earlier one live rolls the bot back to it.")
    for manifest in library.list_releases():
        changed = manifest["changed"]
        entry = manifest["documents"][changed] if changed else None
        title = f"{DOC_LABELS[changed]} → {entry['version']}" if changed else "Initial documents"
        details = [f"published {manifest['published']}"]
        if entry and entry["effective_date"]:
            details.append(f"effective from {entry['effective_date']}")
        if manifest.get("note"):
            details.append(manifest["note"])
        with st.container(border=True):
            left, right = st.columns([5, 1])
            left.markdown(f"**{title}**  \n<small>{' · '.join(details)}</small>", unsafe_allow_html=True)
            if manifest["release_id"] == live_id:
                right.markdown("🟢 **Live**")
            elif right.button("Make live", key=f"live-{manifest['release_id']}"):
                library.make_live(manifest["release_id"])
                flash(f"{title} is live again.")
                st.rerun()


require_password()
init_db()
df = load_messages()

st.title("Kasturi support · admin")
st.caption("Every customer question the bot handled, what it couldn't answer, and what people ask about.")

if df.empty:
    st.info("No conversations yet. Ask the bot something at http://localhost:8000, "
            "or run `python3 scripts/seed_demo.py` to send a set of sample questions.")
    open_flags = 0
else:
    open_flags = int(((df["needs_human"] == 1) & (df["resolved"] == 0)).sum())
    answered = 1 - df["needs_human"].mean()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Questions", len(df), border=True)
    k2.metric("Conversations", df["session_id"].nunique(), border=True)
    k3.metric("Answered by the bot", f"{answered:.0%}", border=True)
    k4.metric("Waiting for a human", open_flags, border=True)

needs_tab, convo_tab, insight_tab, docs_tab = st.tabs(
    [f"🚩 Needs a human ({open_flags})", "💬 Conversations", "📊 Insights", "📄 Documents"])
with needs_tab:
    needs_human_tab(df)
with convo_tab:
    conversations_tab(df)
with insight_tab:
    if df.empty:
        st.caption("Charts appear once customers start asking questions.")
    else:
        insights_tab(df)
with docs_tab:
    documents_tab()
