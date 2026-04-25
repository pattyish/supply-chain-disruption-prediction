from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def generate_shipments(n: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    shipping_pressure = rng.uniform(0.2, 5.0, size=n)
    port_wait_time = np.clip(rng.gamma(shape=2.5, scale=8.0, size=n), 0.0, 120.0)
    weather_risk = rng.integers(0, 3, size=n)
    distance = rng.uniform(120.0, 4200.0, size=n)

    latent = (
        -2.2
        + 0.55 * shipping_pressure
        + 0.028 * port_wait_time
        + 0.48 * weather_risk
        + 0.22 * (distance / 1000.0)
    )
    delay_prob = 1.0 / (1.0 + np.exp(-latent))
    delayed = rng.binomial(1, np.clip(delay_prob, 0.01, 0.99), size=n)

    # Delay hours are zero for on-time shipments, positive skew for delayed shipments.
    delay_hours = delayed * np.clip(
        rng.lognormal(mean=1.4 + 0.2 * weather_risk, sigma=0.45, size=n),
        0.0,
        96.0,
    )

    return pd.DataFrame(
        {
            "shipping_pressure": shipping_pressure.round(3),
            "port_wait_time": port_wait_time.round(3),
            "weather_risk": weather_risk,
            "distance": distance.round(3),
            "delayed": delayed,
            "delay_hours": delay_hours.round(3),
        }
    )


def generate_routes(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    lane_defs = [
        ("Los Angeles", "Chicago"),
        ("Seattle", "Dallas"),
        ("Houston", "Atlanta"),
        ("New York", "Miami"),
    ]

    route_idx = 1
    for origin, destination in lane_defs:
        for _ in range(4):
            distance_km = rng.uniform(900, 3200)
            transit_hours = distance_km / rng.uniform(52, 72)
            base_cost_usd = 1600 + distance_km * rng.uniform(0.8, 1.4)
            reliability = rng.uniform(0.78, 0.97)
            rows.append(
                {
                    "route_id": f"R{route_idx:03d}",
                    "origin": origin,
                    "destination": destination,
                    "distance_km": round(float(distance_km), 2),
                    "transit_hours": round(float(transit_hours), 2),
                    "base_cost_usd": round(float(base_cost_usd), 2),
                    "reliability": round(float(reliability), 4),
                }
            )
            route_idx += 1

    return pd.DataFrame(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    shipments = generate_shipments()
    routes = generate_routes()

    shipments_path = processed_dir / "synthetic_shipments.csv"
    routes_path = processed_dir / "synthetic_routes.csv"

    shipments.to_csv(shipments_path, index=False)
    routes.to_csv(routes_path, index=False)

    print(f"Wrote {shipments_path}")
    print(f"Wrote {routes_path}")


if __name__ == "__main__":
    main()
