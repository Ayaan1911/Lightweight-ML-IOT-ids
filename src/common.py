"""Shared paths, constants, label mapping, thread control."""
import os, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "MERGED_CSV"
RES, MOD, FIG, PROC = (ROOT / d for d in ("results", "models", "figures", "data_processed"))
SEED = 42
CLASSES = ["Benign", "DDoS", "DoS", "Recon", "Web-based", "Brute force", "Spoofing", "Mirai"]

_CAT = {
    "DDoS": ["DDOS-RSTFINFLOOD", "DDOS-PSHACK_FLOOD", "DDOS-SYN_FLOOD", "DDOS-UDP_FLOOD", "DDOS-TCP_FLOOD",
             "DDOS-ICMP_FLOOD", "DDOS-SYNONYMOUSIP_FLOOD", "DDOS-ACK_FRAGMENTATION", "DDOS-UDP_FRAGMENTATION",
             "DDOS-ICMP_FRAGMENTATION", "DDOS-HTTP_FLOOD", "DDOS-SLOWLORIS"],
    "DoS": ["DOS-UDP_FLOOD", "DOS-SYN_FLOOD", "DOS-TCP_FLOOD", "DOS-HTTP_FLOOD"],
    "Recon": ["RECON-HOSTDISCOVERY", "RECON-OSSCAN", "RECON-PORTSCAN", "RECON-PINGSWEEP", "VULNERABILITYSCAN"],
    "Web-based": ["SQLINJECTION", "COMMANDINJECTION", "BACKDOOR_MALWARE", "UPLOADING_ATTACK", "XSS", "BROWSERHIJACKING"],
    "Brute force": ["DICTIONARYBRUTEFORCE"],
    "Spoofing": ["MITM-ARPSPOOFING", "DNS_SPOOFING"],
    "Mirai": ["MIRAI-GREETH_FLOOD", "MIRAI-UDPPLAIN", "MIRAI-GREIP_FLOOD"],
    "Benign": ["BENIGN", "BENIGNTRAFFIC"],
}
LABEL2CAT = {l: c for c, ls in _CAT.items() for l in ls}


def norm_label(s):
    return str(s).strip().upper()


def set_single_thread():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TF_NUM_INTEROP_THREADS", "TF_NUM_INTRAOP_THREADS"):
        os.environ[v] = "1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def log_test_access(model_id):
    with open(ROOT / "test_access.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{model_id}\n")


def jdump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def jload(path):
    return json.loads(Path(path).read_text())
