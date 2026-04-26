from __future__ import annotations

import numpy as np
import pandas as pd
import requests
import streamlit as st

API_BASE = "http://localhost:8000"


def post_json(endpoint: str, payload: dict, timeout: float = 30.0) -> dict:
    last_error: Exception | None = None
    for _ in range(2):
        try:
            resp = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as exc:
            detail = ""
            try:
                body = exc.response.json()
                detail = body.get("detail") or body.get("message") or str(body)
            except Exception:
                detail = str(exc)
            raise RuntimeError(f"API error on {endpoint}: {detail}") from exc
        except requests.exceptions.ReadTimeout as exc:
            last_error = exc
            continue

    raise RuntimeError(f"API request timed out for {endpoint}: {last_error}")


def get_json(endpoint: str, timeout: float = 20.0) -> dict:
    resp = requests.get(f"{API_BASE}{endpoint}", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


st.set_page_config(page_title="Supply Chain Disruption Monitor", layout="wide")
st.title("Supply Chain Disruption Monitor")
st.caption("Predictive monitoring with what-if simulation, optimization, anomaly root-cause, and model provenance.")

with st.sidebar:
    st.header("Shipment Inputs")
    shipping_pressure = st.slider("Shipping Pressure", 0.0, 5.0, 1.2, step=0.1)
    port_wait = st.number_input("Port wait time (hours)", min_value=0.0, max_value=240.0, value=8.0)
    weather_risk = st.selectbox("Weather risk", [0, 1, 2], index=1)
    distance = st.number_input("Distance (km)", min_value=1.0, value=1800.0)
    cost_per_delay_hour = st.number_input("Cost per delay hour (USD)", min_value=0.0, value=80.0)
    sla_penalty = st.number_input("SLA penalty (USD)", min_value=0.0, value=800.0)

    base_payload = {
        "shipping_pressure": float(shipping_pressure),
        "port_wait_time": float(port_wait),
        "weather_risk": int(weather_risk),
        "distance": float(distance),
    }
    impact_payload = {
        **base_payload,
        "cost_per_delay_hour_usd": float(cost_per_delay_hour),
        "sla_penalty_usd": float(sla_penalty),
    }

tab_predict, tab_opt, tab_anomaly, tab_provenance = st.tabs(
    ["Prediction & What-if", "Prescriptive Optimization", "Anomaly + Playbooks", "Provenance + Feedback"]
)

with tab_predict:
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Current prediction")
        if st.button("Run prediction", key="run_pred"):
            try:
                pred = post_json("/predict", base_payload)
                impact = post_json("/impact", impact_payload)
                st.session_state["last_pred"] = pred

                st.metric("Delay Probability", f"{impact['delay_probability']*100:.1f}%")
                st.metric("Risk Level", impact["risk_level"])
                st.progress(min(max(int(impact["delay_probability"] * 100), 0), 100))

                c1, c2, c3 = st.columns(3)
                c1.metric("Expected Delay Hours", f"{impact['expected_delay_hours']:.2f}")
                c2.metric("Expected Delay Cost", f"${impact['expected_delay_cost_usd']:.2f}")
                c3.metric("Expected Total Impact", f"${impact['expected_total_impact_usd']:.2f}")

                st.write(f"Prediction ID: {pred['prediction_id']}")
                st.write(f"Model: {pred['model_name']} ({pred['model_version']})")
            except Exception as e:
                st.error(f"Prediction request failed: {e}")

        st.subheader("Scenario what-if")
        st.write("Simulate an intervention before committing operational changes.")
        what_if_shift = st.slider("Reduce port wait by (hours)", 0.0, 36.0, 6.0, step=1.0)
        if st.button("Compare baseline vs what-if", key="run_whatif"):
            try:
                baseline = post_json("/impact", impact_payload)
                scenario_payload = {
                    **impact_payload,
                    "port_wait_time": max(0.0, impact_payload["port_wait_time"] - what_if_shift),
                }
                scenario = post_json("/impact", scenario_payload)
                delta = baseline["expected_total_impact_usd"] - scenario["expected_total_impact_usd"]

                w1, w2, w3 = st.columns(3)
                w1.metric("Baseline Impact", f"${baseline['expected_total_impact_usd']:.2f}")
                w2.metric("Scenario Impact", f"${scenario['expected_total_impact_usd']:.2f}")
                w3.metric("Savings", f"${delta:.2f}")
            except Exception as e:
                st.error(f"What-if simulation failed: {e}")

    with right:
        st.subheader("Recent trend (sample)")
        days = pd.date_range(end=pd.Timestamp.today(), periods=10)
        rng = np.random.default_rng(11)
        pressure_vals = np.clip(rng.normal(loc=shipping_pressure, scale=0.35, size=10), 0, 5)
        wait_vals = np.clip(rng.normal(loc=port_wait, scale=2.0, size=10), 0, None)
        chart_df = pd.DataFrame({"shipping_pressure": pressure_vals, "port_wait_time": wait_vals}, index=days)
        st.line_chart(chart_df)

with tab_opt:
    st.subheader("Cost-aware reroute optimization")
    c1, c2, c3 = st.columns(3)
    origin = c1.selectbox("Origin", ["Los Angeles", "Seattle", "Houston", "New York"])
    destination = c2.selectbox("Destination", ["Chicago", "Dallas", "Atlanta", "Miami"])
    budget = c3.number_input("Budget cap (USD, optional)", min_value=0.0, value=0.0)

    if st.button("Optimize route", key="optimize"):
        try:
            payload = {
                **impact_payload,
                "origin": origin,
                "destination": destination,
            }
            if budget > 0.0:
                payload["budget_usd"] = float(budget)
            result = post_json("/optimize/reroute", payload)

            if not result.get("budget_feasible", True):
                st.warning(result.get("message", "Budget constraint not feasible; showing unconstrained best route."))

            rec = result["recommended_route"]

            st.markdown("### Recommended Route")
            top1, top2, top3, top4 = st.columns(4)
            top1.metric("Route ID", rec["route_id"])
            top2.metric("Risk Level", rec["risk_level"])
            top3.metric("Delay Probability", f"{rec['delay_probability'] * 100:.1f}%")
            top4.metric("Optimized Total Cost", f"${rec['optimized_total_cost_usd']:,.2f}")

            mid1, mid2, mid3, mid4 = st.columns(4)
            mid1.metric("Base Cost", f"${rec['base_cost_usd']:,.2f}")
            mid2.metric("Expected Impact", f"${rec['expected_total_impact_usd']:,.2f}")
            mid3.metric("Transit Hours", f"{rec['transit_hours']:.2f} h")
            mid4.metric("Reliability", f"{rec['reliability'] * 100:.1f}%")

            st.caption(
                f"Lane: {rec['origin']} -> {rec['destination']} | Distance: {rec['distance_km']:.2f} km"
            )

            st.markdown("### Alternative Routes")
            alt_df = pd.DataFrame(result["alternatives"])
            alt_df = alt_df.rename(
                columns={
                    "route_id": "Route",
                    "origin": "Origin",
                    "destination": "Destination",
                    "distance_km": "Distance (km)",
                    "transit_hours": "Transit (h)",
                    "reliability": "Reliability",
                    "delay_probability": "Delay Prob",
                    "risk_level": "Risk",
                    "base_cost_usd": "Base Cost (USD)",
                    "expected_total_impact_usd": "Expected Impact (USD)",
                    "optimized_total_cost_usd": "Optimized Total (USD)",
                }
            )

            for col in ["Reliability", "Delay Prob"]:
                alt_df[col] = (alt_df[col] * 100).round(2)

            st.dataframe(alt_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Optimization failed: {e}")

with tab_anomaly:
    st.subheader("Anomaly root-cause and playbooks")
    if st.button("Analyze anomaly", key="run_anomaly"):
        try:
            result = post_json("/analyze/anomaly", base_payload)
            st.metric("Anomaly Score", f"{result['anomaly_score']:.3f}")
            st.metric("Risk Level", result["risk_level"])
            st.write(f"Is anomaly: {result['is_anomaly']}")

            st.write("Root causes")
            if result["root_causes"]:
                for cause in result["root_causes"]:
                    st.write(f"- {cause}")
            else:
                st.write("- No strong out-of-pattern features detected.")

            st.write("Playbook actions")
            for action in result["playbook_actions"]:
                st.write(f"- {action}")

            st.write("Feature z-scores")
            z_df = pd.DataFrame([result["feature_zscores"]]).T.reset_index()
            z_df.columns = ["feature", "zscore"]
            st.dataframe(z_df)
        except Exception as e:
            st.error(f"Anomaly analysis failed: {e}")

with tab_provenance:
    st.subheader("Model provenance and active learning")
    p1, p2 = st.columns([2, 3])
    with p1:
        if st.button("Refresh provenance", key="refresh_provenance"):
            try:
                st.session_state["provenance"] = get_json("/provenance")
            except Exception as e:
                st.error(f"Could not fetch provenance: {e}")

        prov = st.session_state.get("provenance")
        if prov:
            st.write("Model")
            st.json(prov["model"])
            st.write("Active learning")
            st.json(prov["active_learning"])

    with p2:
        st.write("Submit feedback label")
        prediction_id = st.text_input(
            "Prediction ID",
            value=st.session_state.get("last_pred", {}).get("prediction_id", ""),
        )
        actual_delayed = st.selectbox("Actual delayed", [0, 1], index=0)
        actual_delay_hours = st.number_input("Actual delay hours", min_value=0.0, value=0.0)
        notes = st.text_area("Notes")

        if st.button("Submit feedback", key="submit_feedback"):
            if not prediction_id:
                st.warning("Prediction ID is required.")
            else:
                try:
                    feedback_payload = {
                        "prediction_id": prediction_id,
                        "actual_delayed": int(actual_delayed),
                        "actual_delay_hours": float(actual_delay_hours),
                        "notes": notes,
                    }
                    result = post_json("/feedback", feedback_payload)
                    st.success(result["message"])
                except Exception as e:
                    st.error(f"Feedback submission failed: {e}")
