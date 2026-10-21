"""Synthetic application testbed.

Applications are generated from a vector of attributes and rendered to text by a
deterministic template renderer. Every random choice that shapes the wording is drawn
from a per-application "style" stream in a fixed order, independent of attribute values.
Re-rendering an application with one attribute changed therefore produces a minimal
edit: only the sentences that express that attribute change.

No real application text is used anywhere. Attribute distributions are stylised and
should be replaced by the real marginals (sector, city, team size, founder gender)
before the testbed is used as a stand-in for the XSEED data.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
FEMALE_NAMES = ["Ayesha", "Fatima", "Sana", "Hira", "Maryam", "Zainab", "Amna", "Mahnoor",
                "Rabia", "Saima", "Nida", "Iqra", "Sadia", "Komal", "Mehwish", "Anum",
                "Bushra", "Farah", "Kiran", "Uzma"]
MALE_NAMES = ["Ahmed", "Ali", "Usman", "Bilal", "Hamza", "Imran", "Faisal", "Kamran",
              "Zubair", "Asad", "Danish", "Haris", "Junaid", "Omar", "Saad", "Talha",
              "Waqas", "Yasir", "Shahid", "Adnan"]
LAST_NAMES = ["Khan", "Farooqi", "Malik", "Qureshi", "Siddiqui", "Shaikh", "Hussain", "Raza",
              "Butt", "Chaudhry", "Mirza", "Abbasi", "Baig", "Memon", "Rizvi"]

CITIES = {"metro": ["Karachi", "Lahore", "Islamabad", "Rawalpindi"],
          "secondary": ["Hyderabad", "Multan", "Peshawar", "Quetta"]}

# Two sector clusters of five sectors each; index i in one cluster is the counterpart
# of index i in the other, which defines the sector edit operator.
SECTORS = {
    "consumer": [
        dict(name="fashion", product="a made-to-order clothing brand that sells stitched kurtas online",
             problem="Small tailors cannot reach buyers outside their neighbourhood, and buyers struggle to find reliable stitched clothing at fair prices.",
             customer="young professionals who buy clothing online",
             details=["We surveyed 40 tailoring shops in {city} and 31 of them had no online presence.",
                      "Our unit cost for a stitched kurta is PKR 1,150 and the selling price is PKR 2,400.",
                      "Repeat orders made up 35 percent of our test sales."],
             trade="garment", field="textiles"),
        dict(name="beauty", product="a line of locally made herbal skincare products sold through social media",
             problem="Imported skincare is expensive and often unsuited to the local climate.",
             customer="urban consumers aged 18 to 35 who shop on social media",
             details=["We tested three formulations with 60 users and kept the one rated best by 70 percent.",
                      "Each jar costs PKR 310 to produce and sells for PKR 950.",
                      "Our Instagram page reached 12,000 followers in four months."],
             trade="cosmetics", field="retail"),
        dict(name="food", product="a home-style frozen meals service that delivers to offices",
             problem="Office workers in {city} lack affordable and hygienic lunch options.",
             customer="office workers and small companies",
             details=["We interviewed 55 office workers and 41 said they skip lunch at least twice a week.",
                      "A meal costs PKR 260 to prepare and sells for PKR 480.",
                      "Two offices have asked for a weekly subscription after our trial week."],
             trade="catering", field="food service"),
        dict(name="handicrafts", product="an online marketplace for hand-embroidered home textiles made by rural artisans",
             problem="Artisans earn very little because middlemen capture most of the margin.",
             customer="urban households and gift buyers",
             details=["We work with 25 artisans from four villages near {city}.",
                      "Artisans receive 45 percent of the sale price, compared with about 15 percent through middlemen.",
                      "Our first batch of 80 cushion covers sold out in three weeks."],
             trade="handicraft", field="crafts"),
        dict(name="home decor", product="a furniture and home decor brand that uses reclaimed wood",
             problem="New furniture is expensive and much usable wood is wasted.",
             customer="young families furnishing their first homes",
             details=["We sourced reclaimed wood from six demolition sites in {city}.",
                      "Our coffee table costs PKR 9,000 to make and sells for PKR 18,500.",
                      "Interior designers have placed 14 orders so far."],
             trade="furniture", field="interior design"),
    ],
    "tech": [
        dict(name="software", product="a cloud-based inventory app for small retailers",
             problem="Small retailers track stock on paper and lose sales when items run out.",
             customer="small retail shops and pharmacies",
             details=["We mapped stock-keeping practices in 45 shops in {city}.",
                      "The app costs PKR 90 per shop per month to run and we plan to charge PKR 1,500.",
                      "Shops in our beta reduced stock-outs by 30 percent."],
             trade="electronics", field="software development"),
        dict(name="healthtech", product="a telemedicine platform that connects patients in smaller towns with qualified doctors",
             problem="Patients outside large cities travel for hours to see a specialist.",
             customer="patients and families in smaller towns",
             details=["We signed up 18 doctors who offer evening consultation slots.",
                      "A consultation costs PKR 400 to deliver and is priced at PKR 900.",
                      "Our pilot handled 320 consultations in two months."],
             trade="pharmacy", field="healthcare"),
        dict(name="agritech", product="a mobile service that sends farmers crop prices and weather alerts by SMS",
             problem="Farmers sell at low prices because they lack timely market information.",
             customer="smallholder farmers and their cooperatives",
             details=["We surveyed 70 farmers near {city} and 52 relied on middlemen for prices.",
                      "Each subscriber costs PKR 35 per month to serve and pays PKR 150.",
                      "Subscribers in our trial sold wheat at 8 percent higher prices."],
             trade="agricultural supplies", field="agriculture"),
        dict(name="edtech", product="an online tutoring platform for secondary-school students preparing for board exams",
             problem="Good tutors are scarce and expensive outside large cities.",
             customer="secondary-school students and their parents",
             details=["We tested our lessons with 90 students in {city}.",
                      "A monthly plan costs PKR 600 to deliver and sells for PKR 2,000.",
                      "Students in our trial improved their mock-exam scores by 11 percent."],
             trade="bookshop", field="education"),
        dict(name="fintech", product="a digital bookkeeping and micro-credit app for shopkeepers",
             problem="Shopkeepers cannot get loans because they keep no formal records.",
             customer="shopkeepers and small traders",
             details=["We interviewed 60 shopkeepers in {city} and 48 had been refused a bank loan.",
                      "Our processing cost is PKR 120 per loan and we earn a 3 percent fee.",
                      "A partner microfinance bank has agreed to a pilot of 100 loans."],
             trade="wholesale trading", field="banking"),
    ],
}

CONSUMER_WORDS = ["clothing", "kurta", "tailor", "skincare", "herbal", "meals", "lunch", "artisans",
                  "embroidered", "furniture", "decor", "reclaimed wood", "cushion"]
TECH_WORDS = ["app", "software", "telemedicine", "doctors", "sms", "farmers", "tutoring",
              "students", "micro-credit", "bookkeeping", "platform", "inventory"]

STAGES = ["We are still at the idea stage and are validating demand.",
          "We have built a working prototype and are testing it with early users.",
          "We have started generating revenue."]
VAGUE = ["We believe there is a very large market for this idea.",
         "Many people need this and nobody is offering it properly.",
         "We are confident that customers will like our product."]

PRONOUNS = {"F": dict(subj="she", obj="her", poss="her", has="has", is_="is"),
            "M": dict(subj="he", obj="him", poss="his", has="has", is_="is"),
            "X": dict(subj="they", obj="them", poss="their", has="have", is_="are")}

N_STYLE = 96  # number of pre-drawn style uniforms per application


# ---------------------------------------------------------------------------
# Application object
# ---------------------------------------------------------------------------
@dataclass
class Application:
    app_id: int
    attrs: dict
    style: np.ndarray = field(repr=False)

    def with_attrs(self, **changes) -> "Application":
        a = copy.deepcopy(self.attrs)
        a.update(changes)
        return Application(self.app_id, a, self.style)

    @property
    def text(self) -> str:
        return render(self)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sample_applications(n: int = 156, seed: int = 0, p_female: float = 0.4) -> list[Application]:
    """Draw n applications with correlated attributes.

    Built-in correlations (the confounds the loop has to get past):
    founder gender -> sector cluster and home-based; city tier -> fluency and foreign degree;
    latent quality -> stage and traction.
    """
    rng = np.random.default_rng(seed)
    apps = []
    for i in range(n):
        g = "F" if rng.random() < p_female else "M"
        tier = "metro" if rng.random() < 0.6 else "secondary"
        consumer = rng.random() < (0.65 if g == "F" else 0.3)
        q = float(rng.normal())
        s = q + rng.normal(0, 0.8)
        stage = 0 if s < -0.4 else (1 if s < 0.6 else 2)
        foreign = rng.random() < (0.3 if tier == "metro" else 0.1)
        fluency = rng.random() < ((0.75 if tier == "metro" else 0.4) + (0.1 if foreign else 0))
        attrs = dict(
            gender=g,
            name_idx=int(rng.integers(len(FEMALE_NAMES))),
            last_idx=int(rng.integers(len(LAST_NAMES))),
            tier=tier,
            city_idx=int(rng.integers(4)),
            cluster="consumer" if consumer else "tech",
            sector_idx=int(rng.integers(5)),
            quality=q,
            stage=stage,
            traction=bool(rng.random() < _sigmoid(-0.8 + 0.9 * q + 0.6 * (stage == 2))),
            fluency_high=bool(fluency),
            foreign_degree=bool(foreign),
            family_business=bool(rng.random() < 0.3),
            home_based=bool(rng.random() < _sigmoid(-1.8 + 1.8 * (g == "F") + 0.6 * consumer)),
            team_size=1 if rng.random() < 0.35 else int(rng.integers(2, 5)),
            experience_years=int(rng.integers(6, 11)) if rng.random() < 0.4 else int(rng.integers(1, 5)),
        )
        style = rng.random(N_STYLE)
        apps.append(Application(i, attrs, style))
    return apps


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _pick(options, u):
    return options[int(u * len(options)) % len(options)]


_PROPER = set(FEMALE_NAMES + MALE_NAMES + CITIES["metro"] + CITIES["secondary"])


def _slip(sentence: str, u: float) -> str:
    """Deterministic, detectable writing slips used for low fluency."""
    s = sentence
    first = s.split(" ", 1)[0].strip(",.")
    proper = first in _PROPER or first.startswith("[") or first == "PKR"
    if s and s[0].isupper() and not proper:
        s = s[0].lower() + s[1:]
    if u < 0.33:
        s = re.sub(r"\b(the|a|an) ", "", s, count=1)
    elif u < 0.66:
        s = s.replace(" has ", " have ", 1).replace(" is ", " are ", 1)
    else:
        s = s.rstrip(".")
    return s


def render_sections(app: Application) -> dict:
    a, u = app.attrs, app.style
    g = a["gender"]
    p = PRONOUNS[g]
    if g == "X":
        first = "[Founder]"; full = "[Founder]"
    else:
        first = (FEMALE_NAMES if g == "F" else MALE_NAMES)[a["name_idx"]]
        full = f"{first} {LAST_NAMES[a['last_idx']]}"
    city = "[City]" if a["tier"] == "X" else CITIES[a["tier"]][a["city_idx"]]
    sec = SECTORS[a["cluster"]][a["sector_idx"]]
    fmt = lambda s: s.format(city=city)

    # Each sentence carries a fixed slot id. Slip decisions for low fluency are keyed to the
    # slot, not to the sentence's position, so adding or removing one sentence never changes
    # how the others are written.
    # --- business idea
    intro = _pick([f"Based in {city}, we are building {sec['product']}.",
                   f"We are a {city}-based startup developing {sec['product']}.",
                   f"Our venture, founded in {city}, offers {sec['product']}."], u[0])
    idea = [(0, intro), (1, STAGES[a["stage"]])]

    # --- problem and evidence (quality controls how specific the text is)
    n_details = int(np.clip(round(1.2 + a["quality"]), 0, 3))
    problem = [(2, fmt(sec["problem"]))] + [(3 + k, fmt(d)) for k, d in enumerate(sec["details"][:n_details])]
    if n_details == 0:
        problem.append((6, _pick(VAGUE, u[1])))
    if a["traction"]:
        n_cust = 6 + int(u[2] * 20)
        problem.append((7, _pick([f"We have already run a pilot with {n_cust} paying customers in {city}.",
                                  f"So far {n_cust} paying customers in {city} have used our pilot."], u[3])))

    # --- target customer
    customer = [(8, _pick([f"Our target customers are {sec['customer']} in {city} and nearby areas.",
                           f"We serve {sec['customer']}, starting in {city}."], u[4]))]
    if n_details <= 1:
        customer.append((9, _pick(VAGUE, u[5])))

    # --- team
    team = []
    if a["team_size"] == 1:
        team.append((10, _pick([f"{full} is the sole founder.", f"The sole founder is {full}."], u[6])))
    else:
        k = ["one", "two", "three"][a["team_size"] - 2]
        noun = "co-founder" if a["team_size"] == 2 else "co-founders"
        team.append((10, _pick([f"{full} leads a team with {k} {noun}.",
                                f"The venture is led by {full}, working with {k} {noun}."], u[6])))
    yrs = a["experience_years"]
    team.append((11, f"{p['subj'].capitalize()} {p['has']} {yrs} years of experience in {sec['field']}."))
    if a["foreign_degree"]:
        team.append((12, f"{p['subj'].capitalize()} {p['has']} a master's degree from a university in the United Kingdom."))
    else:
        team.append((12, f"{p['subj'].capitalize()} {p['has']} a bachelor's degree from a local university."))
    if a["family_business"]:
        team.append((13, f"{first} grew up helping in the family {sec['trade']} business."))
    if a["home_based"]:
        team.append((14, f"The business currently runs from {p['poss']} home."))

    sections = {"Business idea": idea, "Problem to solve": problem,
                "Target customer": customer, "Team": team}
    out = {}
    for key, items in sections.items():
        sents = []
        for slot, s in items:
            # the opening sentence (slot 0) is always slipped so low fluency is always visible
            if not a["fluency_high"] and (slot == 0 or u[50 + slot] < 0.6):
                s = _slip(s, u[10 + slot])
            sents.append(s)
        out[key] = sents
    sections = out
    return {key: " ".join(v) for key, v in sections.items()}


def render(app: Application) -> str:
    return "\n\n".join(f"{k}: {v}" for k, v in render_sections(app).items())


# ---------------------------------------------------------------------------
# Detectors: features as an evaluator (or a coder) would read them from the text
# ---------------------------------------------------------------------------
_FEMALE_RE = re.compile(r"\b(" + "|".join(FEMALE_NAMES) + r")\b")
_MALE_RE = re.compile(r"\b(" + "|".join(MALE_NAMES) + r")\b")


def detect(feature: str, text: str):
    t = text
    tl = text.lower()
    if feature == "gender_cue":  # 1 = female cue, 0 = male cue, None = no cue
        f = bool(_FEMALE_RE.search(t)) or bool(re.search(r"\b(she|her)\b", tl))
        m = bool(_MALE_RE.search(t)) or bool(re.search(r"\b(he|his|him)\b", tl))
        if f and not m:
            return 1
        if m and not f:
            return 0
        return None
    if feature == "consumer_sector":
        c = sum(tl.count(w) for w in CONSUMER_WORDS)
        k = sum(len(re.findall(r"\b" + re.escape(w) + r"\b", tl)) for w in TECH_WORDS)
        return int(c > k)
    if feature == "metro_city":
        if any(c in t for c in CITIES["metro"]):
            return 1
        if any(c in t for c in CITIES["secondary"]):
            return 0
        return None
    if feature == "fluency_high":
        starts = re.findall(r"(?:^|[.:]\s+|\n)([a-z])", t)
        return int(len(starts) == 0)
    if feature == "traction":
        return int(bool(re.search(r"paying (customers|clients|users)|customers (who )?(have )?(already )?paid|paid (for )?(the |our |a )?pilot", tl)))
    if feature == "family_business":
        return int("family" in tl and "business" in tl)
    if feature == "foreign_degree":
        return int(bool(re.search(r"united kingdom|\buk\b|abroad|overseas|foreign university|\blondon\b|british university", tl)))
    if feature == "solo_founder":
        return int(bool(re.search(r"sole founder|solo founder|only founder|single founder|founding the (company|venture|business) alone|without co-?founders", tl)))
    if feature == "home_based":
        return int(bool(re.search(r"\b(from|out of|at) (her |his |their |our |the founder's )?home\b|home-based", tl)))
    if feature == "experienced":
        m = re.search(r"(\d+) years of experience", tl)
        return int(bool(m) and int(m.group(1)) >= 5)
    raise KeyError(feature)


# ---------------------------------------------------------------------------
# Edit operators: one binary feature each, toggled by re-rendering
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Operator:
    name: str
    description: str          # natural-language statement of the feature (value = 1)

    def value(self, app: Application) -> int:
        a = app.attrs
        return {
            "gender_cue": int(a["gender"] == "F"),
            "consumer_sector": int(a["cluster"] == "consumer"),
            "metro_city": int(a["tier"] == "metro"),
            "fluency_high": int(a["fluency_high"]),
            "traction": int(a["traction"]),
            "family_business": int(a["family_business"]),
            "foreign_degree": int(a["foreign_degree"]),
            "solo_founder": int(a["team_size"] == 1),
            "home_based": int(a["home_based"]),
            "experienced": int(a["experience_years"] >= 5),
        }[self.name]

    def set(self, app: Application, v: int) -> Application:
        n = self.name
        if n == "gender_cue":
            return app.with_attrs(gender="F" if v else "M")
        if n == "consumer_sector":
            return app.with_attrs(cluster="consumer" if v else "tech")
        if n == "metro_city":
            return app.with_attrs(tier="metro" if v else "secondary")
        if n == "fluency_high":
            return app.with_attrs(fluency_high=bool(v))
        if n == "traction":
            return app.with_attrs(traction=bool(v))
        if n == "family_business":
            return app.with_attrs(family_business=bool(v))
        if n == "foreign_degree":
            return app.with_attrs(foreign_degree=bool(v))
        if n == "solo_founder":
            if v:
                return app.with_attrs(team_size=1)
            return app.with_attrs(team_size=app.attrs["team_size"] if app.attrs["team_size"] > 1 else 3)
        if n == "home_based":
            return app.with_attrs(home_based=bool(v))
        if n == "experienced":
            if self.value(app) == v:
                return app
            return app.with_attrs(experience_years=8 if v else 2)
        raise KeyError(n)


OPERATORS = [
    Operator("gender_cue", "the founder is identifiable as a woman (name and pronouns)"),
    Operator("consumer_sector", "the venture is in a consumer or lifestyle sector rather than technology"),
    Operator("metro_city", "the venture is based in a large metropolitan city"),
    Operator("fluency_high", "the application is written in fluent, error-free English"),
    Operator("traction", "the application reports paying pilot customers"),
    Operator("family_business", "the founder grew up in a family business"),
    Operator("foreign_degree", "the founder holds a degree from a foreign university"),
    Operator("solo_founder", "the founder applies without co-founders"),
    Operator("home_based", "the business is run from home"),
    Operator("experienced", "the founder has five or more years of relevant experience"),
]
OPS = {o.name: o for o in OPERATORS}


def redact_gender(app: Application) -> Application:
    return app.with_attrs(gender="X")


def redact_geo(app: Application) -> Application:
    return app.with_attrs(tier="X")
