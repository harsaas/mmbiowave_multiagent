from pathlib import Path
import sys

import streamlit as st

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.simulation import LiveSimulation, load_scenario, process_window, window_starts


@st.fragment(run_every=2)
def live_dashboard():
    source = st.session_state.live_source
    controls = st.columns(3)
    if controls[0].button("Pause" if st.session_state.live_running else "Start", key="live_toggle"):
        st.session_state.live_running = not st.session_state.live_running
    step = controls[1].button("Advance 2 simulated seconds", disabled=st.session_state.live_running)
    if controls[2].button("Reset live demo"):
        st.session_state.live_source = LiveSimulation(source.demographic)
        st.session_state.live_history = []
        st.session_state.live_running = False
        st.rerun()
    if source.demographic == "infant":
        if st.button("Restore breathing" if source.apnea else "Inject apnea", key="apnea_toggle"):
            source.apnea = not source.apnea
        st.caption("Injected breathing suppression stays active until restored." if source.apnea
                   else "Synthetic infant breathing is active.")
    else:
        if st.button("Inject fall", disabled=source.pending_fall):
            source.pending_fall = True
        if source.pending_fall:
            st.info("Fall spike queued for the next signal update.")
    st.caption("Initial update initializes 10 seconds; subsequent updates add 2 seconds. "
               "Pause stops advancement. Detection can lag while the rolling window clears.")
    if st.session_state.live_running or step:
        try:
            result = source.advance()
        except ValueError as error:
            st.session_state.live_running = False
            st.error(f"Simulation stopped: {error}")
        else:
            st.session_state.live_history.append(result)
            st.session_state.live_history = st.session_state.live_history[-100:]
    if source.previous is None:
        st.info("Start streaming or advance manually to initialize the signal.")
        return
    result = source.previous
    telemetry, evaluation = result["telemetry"], result["evaluation"]
    cols = st.columns(4)
    cols[0].metric("Estimated breathing / min", telemetry["respiration_rate_br"])
    cols[1].metric("Estimated heartbeats / min", telemetry["heart_rate_bpm"])
    cols[2].metric("Undetected respiration (s)", result["no_respiration_seconds"])
    cols[3].metric("Simulated time (s)", source.samples / 20)
    st.subheader(f"Screening status: {evaluation['status']}")
    st.line_chart({"Raw phase displacement": result["raw_chunk"]})
    st.caption("Rolling 10-second synthetic radar window at 20 Hz.")
    with st.expander("Signal agent - telemetry", expanded=True):
        st.json(telemetry)
    with st.expander("Clinical agent - screening findings", expanded=True):
        for finding in evaluation["clinical_findings"]:
            st.write(f"- {finding}")
    with st.expander("Action agent - simulated response", expanded=True):
        st.info(result["action_result"])
    st.subheader("Recent agent history")
    st.dataframe([
        {"Time (s)": r["covered_end"] / 20, "BR": r["telemetry"]["respiration_rate_br"],
         "BPM": r["telemetry"]["heart_rate_bpm"], "Status": r["evaluation"]["status"],
         "Simulated action": r["action_result"]}
        for r in st.session_state.live_history
    ], hide_index=True)


st.set_page_config(page_title="Mock Radar BioAgent", layout="wide")
st.title("Mock Radar BioAgent")
st.warning("Unvalidated demonstration. Not a clinical monitor. All actions are simulated.")
mode = st.radio("Demo mode", ["Live synthetic stream", "Mock file playback"], horizontal=True)
if mode == "Live synthetic stream":
    cohort = st.selectbox("Care scenario", ["Infant Safety", "Elderly Care"])
    demographic = "infant" if cohort == "Infant Safety" else "elderly"
    if st.session_state.get("live_cohort") != demographic:
        st.session_state.live_cohort = demographic
        st.session_state.live_source = LiveSimulation(demographic)
        st.session_state.live_history = []
        st.session_state.live_running = False
    live_dashboard()
    st.stop()

st.session_state.live_running = False
scenario = st.selectbox("Scenario", ["Infant apnea", "Elderly fall"])
demo = "infant" if scenario == "Infant apnea" else "elderly"
filename = "infant_apnea.json" if demo == "infant" else "elderly_fall.json"
path = Path(__file__).resolve().parent.parent / "data" / "simulated" / filename

if st.session_state.get("scenario") != scenario:
    st.session_state.pop("load_error", None)
    st.session_state.scenario = scenario
    st.session_state.results = []
    try:
        st.session_state.signal = load_scenario(str(path), demo)
    except (OSError, ValueError) as error:
        st.session_state.load_error = str(error)

if "load_error" in st.session_state:
    st.error(f"Cannot load scenario: {st.session_state.load_error}")
    if st.button("Retry loading"):
        st.session_state.pop("scenario", None)
        st.rerun()
    st.stop()

signal = st.session_state.signal
starts = window_starts(len(signal))
results = st.session_state.results
left, right = st.columns(2)
if right.button("Reset demo"):
    st.session_state.results = []
    st.rerun()
if left.button("Process next window", disabled=len(results) >= len(starts)):
    try:
        result = process_window(signal, demo, starts[len(results)], results[-1] if results else None)
    except ValueError as error:
        st.error(f"Window processing failed: {error}")
    else:
        results.append(result)

st.progress(len(results) / len(starts), text=f"{len(results)} / {len(starts)} windows processed")
st.caption("Each click advances two seconds after the initial ten-second window. No live radar is connected.")
if not results:
    st.info("Select Process next window to begin the demo.")
else:
    latest = results[-1]
    telemetry = latest["telemetry"]
    evaluation = latest["evaluation"]
    a, b, c = st.columns(3)
    a.metric("Estimated breathing / min", telemetry["respiration_rate_br"])
    b.metric("Estimated heartbeats / min", telemetry["heart_rate_bpm"])
    c.metric("Undetected respiration (s)", latest["no_respiration_seconds"])
    st.subheader(f"Screening status: {evaluation['status']}")
    for finding in evaluation["clinical_findings"]:
        st.write(f"- {finding}")
    st.info(latest["action_result"])
    st.subheader("Current window: raw phase displacement")
    st.line_chart(latest["raw_chunk"])
    st.caption(f"Samples {latest['covered_end'] - 200} through {latest['covered_end'] - 1}, at 20 Hz.")
    st.subheader("Window history")
    st.dataframe([
        {"End time (s)": result["covered_end"] / 20,
         "BR": result["telemetry"]["respiration_rate_br"],
         "BPM": result["telemetry"]["heart_rate_bpm"],
         "Status": result["evaluation"]["status"]}
        for result in results
    ], hide_index=True)
