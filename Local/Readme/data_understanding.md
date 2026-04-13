# Data Understanding — ENSIA Amphitheatre GPS Dataset



## 1. Project Context

This dataset was collected as part of the **ENSIA Amphitheatre GPS** project.  
Students attending lectures were asked to submit their GPS location (averaged over several readings) along with their **amphitheatre seat position**. The goal is to build a machine-learning model that can **predict which amphitheatre a student is in** (or whether they are outside) purely from GPS/device signals.

---


## 2. Column-by-Column Reference

### 2.1 Flat / Scalar Columns

| # | Column | Type | Description |
|---|--------|------|-------------|
| 1 | `ID` | Integer | Auto-incremented primary key. Unique per submission. |
| 2 | `Timestamp` | DateTime (ISO) | **Client-side** timestamp — when the student pressed "Submit" on their device. May differ from server time. |
| 3 | `Year` | String (`"1"`–`"5"`) | Academic year of the student (1st to 5th year at ENSIA). |
| 4 | `Amphi` | String | The amphitheatre where the student claims to be sitting, e.g. `"Amphi 1"`, `"Amphi 7"`, or a custom location like `"1st floor hallway"`. **This is the target label for classification.** |
| 5 | `Module` | String / null | Name of the course/lecture currently in session. Can be empty if the student did not select a module. |
| 6 | `Block` | String / null | Seating block within the amphitheatre. One of: `Left`, `Center`, `Right`. Null if not applicable (e.g., hallway). |
| 7 | `Row` | Integer / null | Seat row number (positive integer). Null if not provided. |
| 8 | `Column` | Integer / null | Seat column number (positive integer). Null if not provided. |
| 9 | `Lat_Mean` | Float | **Mean latitude** computed from multiple GPS readings. Constrained to Algeria bounds: `18.0 ≤ lat ≤ 38.0`. |
| 10 | `Lng_Mean` | Float | **Mean longitude** computed from multiple GPS readings. Constrained to Algeria bounds: `-9.0 ≤ lng ≤ 12.0`. |
| 11 | `Acc_Mean` | Float | **Mean GPS accuracy** (in metres) across all readings. Lower = better. |
| 12 | `Variance` | Float / null | **GPS variance** computed across the raw readings. Indicates signal stability. Null if only one reading was taken. |
| 13 | `IsOutside` | Boolean | `True` if the student explicitly indicated they are **outside** the building (used as a separate class for indoor/outdoor classification). |

---

### 3.2 JSON Columns (stored as JSON strings in the CSV)

These columns are serialised to a JSON string in the CSV. Parse them with `json.loads()` (Python) or `pd.json_normalize()` before analysis.

---

#### `RawReadings` — Array of GPS samples

Each element is one GPS reading taken during the multi-sample collection window.

```json
[
  {
    "latitude": 36.7123,
    "longitude": 3.1745,
    "accuracy": 8.5,
    "timestamp": 1710000000000
  },
  ...
]
```

| Field | Type | Description |
|-------|------|-------------|
| `latitude` | Float | Raw latitude of this single reading |
| `longitude` | Float | Raw longitude of this single reading |
| `accuracy` | Float | Reported accuracy (metres) for this reading |
| `timestamp` | Integer | Unix timestamp in **milliseconds** |

---

#### `Metadata` — Data collection session info

```json
{
  "num_readings": 10,
  "collection_duration_ms": 15000,
  "collection_start": 1710000000000,
  "collection_end":   1710000015000
}
```

| Field | Type | Description |
|-------|------|-------------|
| `num_readings` | Integer | Total number of GPS samples collected |
| `collection_duration_ms` | Integer | Total duration of the GPS collection window (ms) |
| `collection_start` | Integer | Unix ms timestamp when collection started |
| `collection_end` | Integer | Unix ms timestamp when collection ended |

> **Usage tip:** Use `num_readings` as a data quality filter. Submissions with very few readings (e.g., < 3) may be less reliable.

---

#### `NavigatorContext` — Browser/device environment

```json
{
  "platform": "Linux armv81",
  "userAgent": "Mozilla/5.0 ...",
  "language": "fr-FR",
  "connection": "4g",
  "hardwareConcurrency": 8,
  "deviceMemory": 4,
  "maxTouchPoints": 5,
  "vendor": "Google Inc."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `platform` | String | OS/platform string reported by the browser |
| `userAgent` | String | Full browser user-agent string |
| `language` | String | Browser locale setting |
| `connection` | String | Effective network connection type (`4g`, `3g`, `wifi`, etc.) |
| `hardwareConcurrency` | Integer | Number of logical CPU cores |
| `deviceMemory` | Integer | Device RAM in GB (rounded) |
| `maxTouchPoints` | Integer | Number of touch points (indicator of mobile vs. desktop) |
| `vendor` | String | Browser vendor string |


---

#### `ScreenContext` — Display properties

```json
{
  "width": 412,
  "height": 915,
  "colorDepth": 24,
  "pixelRatio": 2.625,
  "orientation": "portrait-primary"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `width` | Integer | Screen width in pixels |
| `height` | Integer | Screen height in pixels |
| `colorDepth` | Integer | Colour depth (bits) |
| `pixelRatio` | Float | Device pixel ratio (higher = high-DPI screen) |
| `orientation` | String | Screen orientation at submission time |

---

#### `NetworkInfo` — Network connection details

```json
{
  "type": "cellular",
  "effectiveType": "4g",
  "downlink": 10.5,
  "rtt": 100,
  "saveData": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | String | Connection type (`wifi`, `cellular`, `none`, etc.) |
| `effectiveType` | String | Effective speed class (`slow-2g`, `2g`, `3g`, `4g`) |
| `downlink` | Float | Estimated downlink speed (Mbps) |
| `rtt` | Integer | Round-trip time estimate (ms) |
| `saveData` | Boolean | Whether the user has data-saver mode on |

---

#### `BatteryStatus` — Device battery at submission time

```json
{
  "charging": false,
  "level": 0.72,
  "chargingTime": null,
  "dischargingTime": 7200
}
```

| Field | Type | Description |
|-------|------|-------------|
| `charging` | Boolean | Whether the device is currently charging |
| `level` | Float | Battery level `0.0` (empty) → `1.0` (full) |
| `chargingTime` | Integer / null | Seconds until full charge; `null` if not charging |
| `dischargingTime` | Integer / null | Seconds until battery empty; `null` if charging |

---


---

