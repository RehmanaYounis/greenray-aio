"""Exam domains and their weights in the official NCP-AIO blueprint."""

DOMAINS: tuple[str, ...] = (
    "Installation and Deployment",
    "Administration",
    "Workload Management",
    "Troubleshooting and Optimization",
)

# Published on the NVIDIA NCP-AIO certification page; re-check before the exam.
DOMAIN_WEIGHTS: dict[str, float] = {
    "Installation and Deployment": 0.31,
    "Administration": 0.23,
    "Workload Management": 0.23,
    "Troubleshooting and Optimization": 0.23,
}

DOMAIN_PREFIX: dict[str, str] = {
    "Installation and Deployment": "ins",
    "Administration": "adm",
    "Workload Management": "wkl",
    "Troubleshooting and Optimization": "tsh",
}
