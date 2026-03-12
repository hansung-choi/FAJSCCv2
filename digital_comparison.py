import numpy as np
import pandas as pd

# -----------------------------
# 1) (model_name, cpp_list, psnr_list) -> per-model dict
# -----------------------------
def build_cpp_psnr_dict(model_name, cpp_list, psnr_list):
    """
    Inputs
      - model_name: str
      - cpp_list: list/tuple/np.ndarray of CPP values (float)
      - psnr_list: list/tuple/np.ndarray of PSNR values (float), same length as cpp_list

    Output (dict)
      - stores which CPPs exist
      - allows access by CPP key: d["by_cpp"][cpp] -> psnr
      - also provides sorted arrays: d["cpp_list_sorted"], d["psnr_list_sorted"]
    """
    if len(cpp_list) != len(psnr_list):
        raise ValueError("cpp_list and psnr_list must have the same length.")

    cpp_arr = np.asarray(cpp_list, dtype=float)
    psnr_arr = np.asarray(psnr_list, dtype=float)

    # Remove NaNs / invalid pairs
    mask = np.isfinite(cpp_arr) & np.isfinite(psnr_arr)
    cpp_arr = cpp_arr[mask]
    psnr_arr = psnr_arr[mask]

    if cpp_arr.size == 0:
        raise ValueError("No valid (cpp, psnr) pairs remain after filtering.")

    # Handle duplicates: keep the best PSNR for the same CPP (common practice)
    by_cpp = {}
    for c, p in zip(cpp_arr.tolist(), psnr_arr.tolist()):
        if c in by_cpp:
            by_cpp[c] = max(by_cpp[c], p)
        else:
            by_cpp[c] = p

    # Sorted arrays
    cpp_sorted = np.array(sorted(by_cpp.keys()), dtype=float)
    psnr_sorted = np.array([by_cpp[c] for c in cpp_sorted], dtype=float)

    out = {
        "model_name": model_name,
        "cpp_set": set(by_cpp.keys()),
        "by_cpp": by_cpp,  # key: cpp -> value: psnr
        "cpp_list_sorted": cpp_sorted,
        "psnr_list_sorted": psnr_sorted,
    }
    return out


# -----------------------------
# BD-metric helpers (Bjontegaard)
# -----------------------------
def _polyfit_safe(x, y, deg=3):
    """
    Fit polynomial with a safe degree:
    - requires at least deg+1 points, otherwise lowers degree to n-1.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 points for BD metric computation.")
    use_deg = min(deg, n - 1)
    return np.polyfit(x, y, use_deg)

def _polyint_eval(coeff, x0, x1):
    """
    Integrate polynomial defined by coeff over [x0, x1].
    coeff: np.polyfit output (highest power first)
    """
    pint = np.polyint(coeff)
    return np.polyval(pint, x1) - np.polyval(pint, x0)

def bd_psnr(cpp_ref, psnr_ref, cpp_test, psnr_test):
    """
    BD-PSNR: average PSNR difference (test - ref) over overlapping log(CPP) interval.
    Fits PSNR = f(log(CPP)).
    Returns: delta PSNR (dB)
    """
    cpp_ref = np.asarray(cpp_ref, dtype=float)
    psnr_ref = np.asarray(psnr_ref, dtype=float)
    cpp_test = np.asarray(cpp_test, dtype=float)
    psnr_test = np.asarray(psnr_test, dtype=float)

    # log-domain CPP (like BD-rate uses log-rate)
    lr_ref = np.log(cpp_ref)
    lr_test = np.log(cpp_test)

    # overlapping interval
    x_min = max(lr_ref.min(), lr_test.min())
    x_max = min(lr_ref.max(), lr_test.max())
    if x_max <= x_min:
        raise ValueError("No overlap in log(CPP) range for BD-PSNR.")

    # fit PSNR vs log(CPP)
    c_ref = _polyfit_safe(lr_ref, psnr_ref, deg=3)
    c_test = _polyfit_safe(lr_test, psnr_test, deg=3)

    int_ref = _polyint_eval(c_ref, x_min, x_max)
    int_test = _polyint_eval(c_test, x_min, x_max)

    avg_ref = int_ref / (x_max - x_min)
    avg_test = int_test / (x_max - x_min)
    return float(avg_test - avg_ref)

def bd_cpp(cpp_ref, psnr_ref, cpp_test, psnr_test):
    """
    BD-CPP (BD-rate analog): average percentage CPP difference (test vs ref)
    over overlapping PSNR interval.

    Fits log(CPP) = g(PSNR).
    Returns: delta CPP in percent (%), i.e., (exp(avg_diff)-1)*100
    """
    cpp_ref = np.asarray(cpp_ref, dtype=float)
    psnr_ref = np.asarray(psnr_ref, dtype=float)
    cpp_test = np.asarray(cpp_test, dtype=float)
    psnr_test = np.asarray(psnr_test, dtype=float)

    lr_ref = np.log(cpp_ref)
    lr_test = np.log(cpp_test)

    # overlapping PSNR interval
    y_min = max(psnr_ref.min(), psnr_test.min())
    y_max = min(psnr_ref.max(), psnr_test.max())
    if y_max <= y_min:
        raise ValueError("No overlap in PSNR range for BD-CPP.")

    # fit log(CPP) vs PSNR
    c_ref = _polyfit_safe(psnr_ref, lr_ref, deg=3)
    c_test = _polyfit_safe(psnr_test, lr_test, deg=3)

    int_ref = _polyint_eval(c_ref, y_min, y_max)
    int_test = _polyint_eval(c_test, y_min, y_max)

    avg_diff = (int_test - int_ref) / (y_max - y_min)  # in log-domain
    return float((np.exp(avg_diff) - 1.0) * 100.0)


# -----------------------------
# 2) total_dict + model list + anchor -> BD-CPP / BD-PSNR table
# -----------------------------
def make_bd_table(total_dictionary, model_name_list, anchor_model_name, save_csv_path=None):
    """
    Inputs
      - total_dictionary: dict[model_name] -> per-model dict (from build_cpp_psnr_dict)
      - model_name_list: list of model names to evaluate (including/excluding anchor ok)
      - anchor_model_name: str, reference model
      - save_csv_path: optional str path to save the table as CSV

    Output
      - pandas.DataFrame (easy to view / save)
    """
    if anchor_model_name not in total_dictionary:
        raise KeyError(f"anchor_model_name '{anchor_model_name}' not found in total_dictionary.")

    anchor = total_dictionary[anchor_model_name]
    cpp_a = anchor["cpp_list_sorted"]
    psnr_a = anchor["psnr_list_sorted"]

    rows = []
    for m in model_name_list:
        if m not in total_dictionary:
            rows.append({
                "Model": m,
                "Anchor": anchor_model_name,
                "BD-CPP (%)": None,
                "BD-PSNR (dB)": None,
                "Note": "missing model in total_dictionary"
            })
            continue

        if m == anchor_model_name:
            rows.append({
                "Model": m,
                "Anchor": anchor_model_name,
                "BD-CPP (%)": 0.0,
                "BD-PSNR (dB)": 0.0,
                "Note": "anchor"
            })
            continue

        d = total_dictionary[m]
        cpp_m = d["cpp_list_sorted"]
        psnr_m = d["psnr_list_sorted"]

        try:
            delta_cpp = bd_cpp(cpp_a, psnr_a, cpp_m, psnr_m)
            delta_psnr = bd_psnr(cpp_a, psnr_a, cpp_m, psnr_m)
            note = ""
        except Exception as e:
            delta_cpp = None
            delta_psnr = None
            note = str(e)

        rows.append({
            "Model": m,
            "Anchor": anchor_model_name,
            "BD-CPP (%)": delta_cpp,
            "BD-PSNR (dB)": delta_psnr,
            "Note": note
        })

    df = pd.DataFrame(rows, columns=["Model", "Anchor", "BD-CPP (%)", "BD-PSNR (dB)", "Note"])

    def _sort_key(v, big=1e18):
        return big if (v is None or (isinstance(v, float) and np.isnan(v))) else v

    df["_bdcpp_sort"] = df["BD-CPP (%)"].apply(_sort_key)
    df = df.sort_values(by=["_bdcpp_sort", "Model"]).drop(columns=["_bdcpp_sort"]).reset_index(drop=True)

    if save_csv_path is not None:
        df.to_csv(save_csv_path, index=False)

    return df




# -----------------------------
# 1) (model_name, cpp_list, msssim_list) -> per-model dict 
# -----------------------------
def build_cpp_msssim_dict(model_name, cpp_list, msssim_list):
    """
    Stores:
      - which CPPs exist
      - access by CPP key: d["by_cpp"][cpp] -> msssim
      - sorted arrays: d["cpp_list_sorted"], d["msssim_list_sorted"]
    """
    if len(cpp_list) != len(msssim_list):
        raise ValueError("cpp_list and msssim_list must have the same length.")

    cpp_arr = np.asarray(cpp_list, dtype=float)
    ms_arr = np.asarray(msssim_list, dtype=float)

    # filter invalid
    mask = np.isfinite(cpp_arr) & np.isfinite(ms_arr)
    cpp_arr = cpp_arr[mask]
    ms_arr = ms_arr[mask]
    if cpp_arr.size == 0:
        raise ValueError("No valid (cpp, ms-ssim) pairs remain after filtering.")

    # handle duplicates: keep the best MS-SSIM for same CPP
    by_cpp = {}
    for c, v in zip(cpp_arr.tolist(), ms_arr.tolist()):
        if c in by_cpp:
            by_cpp[c] = max(by_cpp[c], v)
        else:
            by_cpp[c] = v

    cpp_sorted = np.array(sorted(by_cpp.keys()), dtype=float)
    ms_sorted = np.array([by_cpp[c] for c in cpp_sorted], dtype=float)

    return {
        "model_name": model_name,
        "cpp_set": set(by_cpp.keys()),
        "by_cpp": by_cpp,  # cpp -> ms-ssim
        "cpp_list_sorted": cpp_sorted,
        "msssim_list_sorted": ms_sorted,
    }


# -----------------------------
# BD-metric core utilities
# -----------------------------
def _polyfit_safe(x, y, deg=3):
    """
    Fit polynomial with safe degree:
      - needs >= deg+1 points, otherwise uses degree = n-1
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 points for BD metric computation.")
    use_deg = min(deg, n - 1)
    return np.polyfit(x, y, use_deg)

def _polyint_eval(coeff, a, b):
    pint = np.polyint(coeff)
    return np.polyval(pint, b) - np.polyval(pint, a)


def bd_metric_y(cpp_ref, y_ref, cpp_test, y_test):
    """
    Generic BD-Y: average (y_test - y_ref) over overlapping log(CPP) interval.
    Fits y = f(log(CPP)).
    Returns: delta_y
    """
    cpp_ref = np.asarray(cpp_ref, dtype=float)
    y_ref   = np.asarray(y_ref, dtype=float)
    cpp_t   = np.asarray(cpp_test, dtype=float)
    y_t     = np.asarray(y_test, dtype=float)

    lr_ref = np.log(cpp_ref)
    lr_t   = np.log(cpp_t)

    x_min = max(lr_ref.min(), lr_t.min())
    x_max = min(lr_ref.max(), lr_t.max())
    if x_max <= x_min:
        raise ValueError("No overlap in log(CPP) range.")

    c_ref = _polyfit_safe(lr_ref, y_ref, deg=3)
    c_t   = _polyfit_safe(lr_t,   y_t,   deg=3)

    int_ref = _polyint_eval(c_ref, x_min, x_max)
    int_t   = _polyint_eval(c_t,   x_min, x_max)

    avg_ref = int_ref / (x_max - x_min)
    avg_t   = int_t   / (x_max - x_min)
    return float(avg_t - avg_ref)
    

def bd_cpp(cpp_ref, y_ref, cpp_test, y_test):
    """
    BD-CPP (%): average percentage CPP difference (test vs ref)
    over overlapping Y interval (here Y can be MS-SSIM).
    Fits log(CPP) = g(Y).
    Returns: percent (%)
    """
    cpp_ref = np.asarray(cpp_ref, dtype=float)
    y_ref   = np.asarray(y_ref, dtype=float)
    cpp_t   = np.asarray(cpp_test, dtype=float)
    y_t     = np.asarray(y_test, dtype=float)

    lr_ref = np.log(cpp_ref)
    lr_t   = np.log(cpp_t)

    y_min = max(y_ref.min(), y_t.min())
    y_max = min(y_ref.max(), y_t.max())
    if y_max <= y_min:
        raise ValueError("No overlap in Y (metric) range.")

    c_ref = _polyfit_safe(y_ref, lr_ref, deg=3)
    c_t   = _polyfit_safe(y_t,   lr_t,   deg=3)

    int_ref = _polyint_eval(c_ref, y_min, y_max)
    int_t   = _polyint_eval(c_t,   y_min, y_max)

    avg_diff = (int_t - int_ref) / (y_max - y_min)  # log-domain
    return float((np.exp(avg_diff) - 1.0) * 100.0)


# -----------------------------
# 2) total_dictionary + model list + anchor -> BD-CPP / BD-MS-SSIM table
# -----------------------------
def make_bd_msssim_table(total_dictionary, model_name_list, anchor_model_name, save_csv_path=None):
    """
    Output columns:
      - BD-CPP (%) : average CPP difference (%)
      - BD-MS-SSIM (dB) : average MS-SSIM difference in dB domain
    """

    if anchor_model_name not in total_dictionary:
        raise KeyError(f"anchor_model_name '{anchor_model_name}' not found in total_dictionary.")

    def msssim_to_db(x):
        # numerical stability
        eps = 1e-12
        x = np.clip(x, 0.0, 1.0 - eps)
        return -10.0 * np.log10(1.0 - x)

    anchor = total_dictionary[anchor_model_name]
    cpp_a = anchor["cpp_list_sorted"]
    y_a_db = msssim_to_db(anchor["msssim_list_sorted"])

    rows = []

    for m in model_name_list:
        if m not in total_dictionary:
            rows.append({
                "Model": m,
                "Anchor": anchor_model_name,
                "BD-CPP (%)": None,
                "BD-MS-SSIM (dB)": None,
                "Note": "missing model"
            })
            continue

        if m == anchor_model_name:
            rows.append({
                "Model": m,
                "Anchor": anchor_model_name,
                "BD-CPP (%)": 0.0,
                "BD-MS-SSIM (dB)": 0.0,
                "Note": "anchor"
            })
            continue

        d = total_dictionary[m]
        cpp_m = d["cpp_list_sorted"]
        y_m_db = msssim_to_db(d["msssim_list_sorted"])

        try:
            delta_cpp = bd_cpp(cpp_a, y_a_db, cpp_m, y_m_db)
            delta_db  = bd_metric_y(cpp_a, y_a_db, cpp_m, y_m_db)
            note = ""
        except Exception as e:
            delta_cpp = None
            delta_db  = None
            note = str(e)

        rows.append({
            "Model": m,
            "Anchor": anchor_model_name,
            "BD-CPP (%)": delta_cpp,
            "BD-MS-SSIM (dB)": delta_db,
            "Note": note
        })

    df = pd.DataFrame(rows, columns=["Model", "Anchor", "BD-CPP (%)", "BD-MS-SSIM (dB)", "Note"])

    # sort (smaller BD-CPP is better)
    def _sort_key(v, big=1e18):
        if v is None:
            return big
        try:
            if np.isnan(v):
                return big
        except Exception:
            pass
        return v

    df["_sort"] = df["BD-CPP (%)"].apply(_sort_key)
    df = df.sort_values(by=["_sort", "Model"]).drop(columns=["_sort"]).reset_index(drop=True)

    if save_csv_path is not None:
        df.to_csv(save_csv_path, index=False)

    return df
    
def main():
    # using LDPC + QAM at Kodak
    print("-----------using LDPC + QAM at Kodak-----------")
    HugeFAJSCC = build_cpp_psnr_dict("HugeFAJSCC", cpp_list=[1/12,1/16,1/24,1/32], psnr_list=[34.17,32.95,31.27,30.28])
    LDPC_rate= 2/3 #Rate of LDPC
    QAM = 16
    B = 1/ LDPC_rate / np.log2(QAM) / 3
    JPEG = build_cpp_psnr_dict("JPEG", cpp_list=[0.4231*B,0.5083*B,0.5878*B,0.6601*B], psnr_list=[28.04,29.04,29.78,30.37])
    JPEG2000 = build_cpp_psnr_dict("JPEG2000", cpp_list=[0.7982*B,0.5988*B,0.4792*B,0.3986*B], psnr_list=[33.48,32.08,31.07,30.30])
    BPG = build_cpp_psnr_dict("BPG", cpp_list=[0.0677*B,0.1610*B,0.3515*B,0.6846*B], psnr_list=[26.19,28.68,31.59,34.85])
    VTM = build_cpp_psnr_dict("VTM", cpp_list=[0.0482*B,0.1124*B,0.2458*B,0.4905*B], psnr_list=[26.14,28.49,31.19,34.26])

    # total_dictionary.
    total_dictionary = {
        "JPEG": JPEG,
        "JPEG2000": JPEG2000,
        "BPG": BPG,
        "VTM": VTM,
        "HugeFAJSCC":HugeFAJSCC,
    }

    # 2) BD-CPP / BD-PSNR make table and save.
    df = make_bd_table(
        total_dictionary,
        model_name_list=["JPEG","JPEG2000","BPG","VTM","HugeFAJSCC"],
        anchor_model_name="JPEG2000",
        save_csv_path="./test_results/BD_table_HugeFAJSCC.csv"
    )
    print(df)
    print("0.0482*B:",0.0482*B)

    print("-----------using LDPC + QAM at Kodak-----------")
    print("-------------------MS-SSIM---------------------")
    HugeFAJSCC = build_cpp_msssim_dict("HugeFAJSCC", cpp_list=[1/12,1/16,1/24,1/32], msssim_list=[0.9845,0.9825,0.9735,0.9665])        
    LDPC_rate= 2/3 #Rate of LDPC
    QAM = 16
    B = 1/ LDPC_rate / np.log2(QAM) / 3
    JPEG = build_cpp_msssim_dict("JPEG", cpp_list=[0.4231*B,0.5083*B,0.5878*B,0.6601*B], msssim_list=[0.923,0.942,0.953,0.960])
    JPEG2000 = build_cpp_msssim_dict("JPEG2000", cpp_list=[0.7982*B,0.5988*B,0.4792*B,0.3986*B], msssim_list=[0.970,0.960,0.953,0.944])
    BPG = build_cpp_msssim_dict("BPG", cpp_list=[0.0677*B,0.1610*B,0.3515*B,0.6846*B], msssim_list=[0.868,0.924,0.961,0.980])
    VTM = build_cpp_msssim_dict("VTM", cpp_list=[0.0482*B,0.1124*B,0.2458*B,0.4905*B], msssim_list=[0.860,0.917,0.956,0.978])

    # total_dictionary.
    total_dictionary = {
        "JPEG": JPEG,
        "JPEG2000": JPEG2000,
        "BPG": BPG,
        "VTM": VTM,
        "HugeFAJSCC":HugeFAJSCC,
    }

    # 2) BD-CPP / BD-MS-SSIM make table and save.
    df = make_bd_msssim_table(
        total_dictionary,
        model_name_list=["JPEG","JPEG2000","BPG","VTM","HugeFAJSCC"],
        anchor_model_name="JPEG2000",
        save_csv_path="./test_results/BD_table_MS-SSIM_HugeFAJSCC.csv"
    )
    print(df)
    print("0.0482*B:",0.0482*B)

if __name__ == '__main__':
    main()
    

















