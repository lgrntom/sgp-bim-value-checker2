import io
import re
import unicodedata
import tempfile
from typing import Dict, Any, List, Tuple

import pandas as pd
import streamlit as st
import ifcopenshell

# Utilitaires IfcOpenShell (souvent dispo, sinon fallback)
try:
    import ifcopenshell.util.element as ifcutil_element
except Exception:
    ifcutil_element = None


# =========================
# CONFIG SGP (TES PSET)
# =========================
# NB: Tu peux renommer les Pset si besoin. L'outil est fait pour être modifiable.
PSET_IDENTIFIANT = "Pset_SGP_DI_IDENTIFIANT"
PSET_LOCALISATION_OUVRAGE = "Pset_SGP_DI_LOCALISATION_OUVRAGE"
PSET_LOCALISATION_LIGNE = "Pset_SGP_DI_LOCALISATION_LIGNE"
PSET_UNICITE = "Pset_SGP_DI_UNICITE"
PSET_REFERENCES = "Pset_SGP_DI_REFERENCES"
PSET_CODE_BIM = "Pset_SGP_DI_CODE_BIM"

PSET_INTITULES = "Pset_SGP_DI_INTITULES"
PSET_GDC = "Pset_SGP_DI_GDC"  # optionnel

# Champs obligatoires par Pset (tes listes)
REQ = {
    PSET_IDENTIFIANT: [
        "SGP_DI_IDENTIFIANT",
        "SGP_DI_DOMAINE",
        "SGP_DI_SPECIALITE",
        "SGP_DI_SOUS_SPECIALITE",
        "SGP_DI_TYPOLOGIE_EQUIPEMENT",
        "SGP_DI_GAMME_EQUIPEMENT",
        "SGP_DI_SOUS_COMPOSANTS_01",
        "SGP_DI_SOUS_COMPOSANTS_02",
        "SGP_DI_SOUS_COMPOSANTS_03",
        "SGP_DI_SOUS_COMPOSANTS_04",
        "SGP_DI_SOUS_COMPOSANTS_05",
    ],
    PSET_LOCALISATION_OUVRAGE: [
        "SGP_DI_LOCALISATION",
        "SGP_DI_OUVRAGE",
        "SGP_DI_BATIMENT",
        "SGP_DI_NIVEAU",
        "SGP_DI_TYPOLOGIE_ESPACE",
        "SGP_DI_INCREMENTATION",
    ],
    PSET_LOCALISATION_LIGNE: [
        "SGP_DI_LOCALISATION",
        "SGP_DI_OUVRAGE",
        "SGP_DI_ZONE",
        "SGP_DI_VOIE",
        "SGP_DI_DECOUPAGE_KM",
        "SGP_DI_REPERAGE_METRE",
    ],
    PSET_UNICITE: [
        "SGP_DI_UNICITE",
        "SGP_DI_CODIFICATION",
    ],
    PSET_REFERENCES: [
        "SGP_REF_PROCEDURE_CODIFICATION",
        "SGP_REF_REFERENTIEL_DONNEES",
        "SGP_REF_MARCHE",
        "SGP_REF_OBJET_VERSION",
    ],
    PSET_CODE_BIM: [
        "SGP_DI_CODE_BIM",
    ],
    # Champs "intitulés" : utiles mais on ne les met pas bloquants par défaut.
    # Tu peux les rendre bloquants si tu veux.
    PSET_INTITULES: [
        "SGP_DI_INTITULE_OUVRAGE",
        "SGP_DI_INTITULE_BATIMENT",
        "SGP_DI_INTITULE_NIVEAU",
        "SGP_DI_INTITULE_TYPOLOGIE_ESPACE",
        "SGP_DI_INTITULE_DOMAINE",
        "SGP_DI_INTITULE_SPECIALITE",
        "SGP_DI_INTITULE_SOUS_SPECIALITE",
        "SGP_DI_INTITULE_TYPOLOGIE_EQUIPEMENT",
        "SGP_DI_INTITULE_GAMME_EQUIPEMENT",
        "SGP_DI_INTITULE_SOUS_COMPOSANTS_01",
        "SGP_DI_INTITULE_SOUS_COMPOSANTS_02",
        "SGP_DI_INTITULE_SOUS_COMPOSANTS_03",
    ],
    # GDC optionnel
    PSET_GDC: [
        "SGP_DI_CODE_GDC",
    ],
}

# Psets réellement bloquants pour un DOE patrimonial minimal
BLOCKING_PSETS = {
    PSET_IDENTIFIANT,
    PSET_UNICITE,
    PSET_REFERENCES,
    PSET_CODE_BIM,
    # localisation : OUVRAGE OU LIGNE, géré à part
}

# Règles transverses
SEP = "-"
NON_USED_VALUE = "000"


# =========================
# UTILS
# =========================
def normalize_val(val) -> str | None:
    if val is None:
        return None
    s = str(val).strip()
    if s == "" or s.lower() in {"undefined", "none", "null"}:
        return None
    return s


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def format_issues(value: str, require_dash: bool = False) -> List[str]:
    issues = []
    if value is None:
        return issues
    if value != value.upper():
        issues.append("FORMAT: pas en majuscules")
    if " " in value:
        issues.append("FORMAT: contient des espaces")
    if strip_accents(value) != value:
        issues.append("FORMAT: contient des accents")
    if require_dash and "-" not in value:
        issues.append("FORMAT: séparateur '-' absent")
    return issues


def save_upload_to_temp(uploaded_file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ifc") as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def get_psets(elem) -> Dict[str, Dict[str, Any]]:
    if ifcutil_element is None:
        return {}
    try:
        return ifcutil_element.get_psets(elem) or {}
    except Exception:
        return {}


def get_pset_value(psets: dict, pset_name: str, prop_name: str) -> str | None:
    d = psets.get(pset_name)
    if not isinstance(d, dict):
        return None
    return normalize_val(d.get(prop_name))


def concat_identifiant(psets: dict) -> str | None:
    """IDENTIFIANT = DOMAINE-SPECIALITE-SOUS_SPECIALITE-TYPOLOGIE-GAMME-SC01-SC02-SC03-SC04-SC05"""
    needed = [
        "SGP_DI_DOMAINE",
        "SGP_DI_SPECIALITE",
        "SGP_DI_SOUS_SPECIALITE",
        "SGP_DI_TYPOLOGIE_EQUIPEMENT",
        "SGP_DI_GAMME_EQUIPEMENT",
        "SGP_DI_SOUS_COMPOSANTS_01",
        "SGP_DI_SOUS_COMPOSANTS_02",
        "SGP_DI_SOUS_COMPOSANTS_03",
        "SGP_DI_SOUS_COMPOSANTS_04",
        "SGP_DI_SOUS_COMPOSANTS_05",
    ]
    parts = []
    for p in needed:
        v = get_pset_value(psets, PSET_IDENTIFIANT, p)
        if v is None:
            return None
        parts.append(v)
    return SEP.join(parts)


def concat_localisation(psets: dict) -> Tuple[str | None, str | None]:
    """
    Retourne (localisation_concatee, mode) avec mode = "OUVRAGE" ou "LIGNE"
    """
    # OUVRAGE
    needed_o = ["SGP_DI_OUVRAGE", "SGP_DI_BATIMENT", "SGP_DI_NIVEAU", "SGP_DI_TYPOLOGIE_ESPACE", "SGP_DI_INCREMENTATION"]
    parts_o = []
    ok_o = True
    for p in needed_o:
        v = get_pset_value(psets, PSET_LOCALISATION_OUVRAGE, p)
        if v is None:
            ok_o = False
            break
        parts_o.append(v)
    if ok_o:
        return (SEP.join(parts_o), "OUVRAGE")

    # LIGNE
    needed_l = ["SGP_DI_OUVRAGE", "SGP_DI_ZONE", "SGP_DI_VOIE", "SGP_DI_DECOUPAGE_KM", "SGP_DI_REPERAGE_METRE"]
    parts_l = []
    ok_l = True
    for p in needed_l:
        v = get_pset_value(psets, PSET_LOCALISATION_LIGNE, p)
        if v is None:
            ok_l = False
            break
        parts_l.append(v)
    if ok_l:
        return (SEP.join(parts_l), "LIGNE")

    return (None, None)


def is_unicite_valid(u: str | None) -> bool:
    """0001 à 9999 attendu. On accepte \d{4} et on refuse 0000."""
    if u is None:
        return False
    if not re.match(r"^\d{4}$", u):
        return False
    return u != "0000"


# =========================
# CORE CHECK
# =========================
def analyze_ifc(model) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Retourne:
      - summary_df: KPI par classe IFC
      - details_df: état par objet
      - issues_df: liste détaillée des anomalies
      - kpi_global: dictionnaire de KPI globaux (taux)
    """
    # On contrôle "toutes classes IFC" => tous les IfcProduct (incluant équipements, objets, etc.)
    elems = model.by_type("IfcProduct")

    rows_details = []
    rows_issues = []
    rows_codes = []

    total = 0
    ok_completeness = 0
    ok_format = 0
    ok_coherence = 0

    for e in elems:
        # skip objets sans GlobalId (rare)
        gid = getattr(e, "GlobalId", None)
        if gid is None:
            continue

        ifc_class = e.is_a()
        total += 1

        psets = get_psets(e)

        element_issues = []
        blocking_issues = []

        # 1) Complétude des Psets bloquants
        for pset_name in BLOCKING_PSETS:
            for prop in REQ.get(pset_name, []):
                v = get_pset_value(psets, pset_name, prop)
                if v is None:
                    blocking_issues.append(f"MANQUANT: {pset_name}.{prop}")
                else:
                    # format transverse
                    need_dash = prop in {"SGP_DI_IDENTIFIANT", "SGP_DI_LOCALISATION", "SGP_DI_CODE_BIM"}
                    element_issues += [f"{pset_name}.{prop} -> {x}" for x in format_issues(v, require_dash=need_dash)]

        # 2) Localisation: au moins un des deux Psets doit être complet
        loc_concat, loc_mode = concat_localisation(psets)
        if loc_concat is None:
            blocking_issues.append("MANQUANT: Localisation (OUVRAGE ou LIGNE) incomplète")
        else:
            element_issues += [f"LOCALISATION({loc_mode}) -> {x}" for x in format_issues(loc_concat, require_dash=True)]

        # 3) Cohérence IDENTIFIANT (valeur calculée vs champ)
        ident_calc = concat_identifiant(psets)
        ident_field = get_pset_value(psets, PSET_IDENTIFIANT, "SGP_DI_IDENTIFIANT")
        if ident_calc is not None and ident_field is not None:
            if ident_field != ident_calc:
                element_issues.append(f"COHERENCE: SGP_DI_IDENTIFIANT incohérent (attendu {ident_calc})")

        # 4) Unicité format
        unicite = get_pset_value(psets, PSET_UNICITE, "SGP_DI_UNICITE")
        if not is_unicite_valid(unicite):
            element_issues.append("FORMAT: SGP_DI_UNICITE attendu sur 4 chiffres (0001-9999), 0000 interdit")

        # 5) Cohérence CODE_BIM = LOC-UNICITE-IDENT
        code_bim = get_pset_value(psets, PSET_CODE_BIM, "SGP_DI_CODE_BIM")
        if loc_concat and unicite and ident_field and code_bim:
            expected = f"{loc_concat}-{unicite}-{ident_field}"
            if code_bim != expected:
                blocking_issues.append(f"COHERENCE: CODE_BIM incohérent (attendu {expected})")
        else:
            # si on n'a pas assez d'info, c'est déjà couvert par "manquants" plus haut
            pass

        # 6) Règle NON UTILISÉ = 000 (on le contrôle sur les sous-composants)
        for prop in ["SGP_DI_SOUS_COMPOSANTS_04", "SGP_DI_SOUS_COMPOSANTS_05"]:
            v = get_pset_value(psets, PSET_IDENTIFIANT, prop)
            if v is None:
                blocking_issues.append(f"MANQUANT: {PSET_IDENTIFIANT}.{prop} (attendu '000' si non utilisé)")

        # Regroupe issues
        all_issues = blocking_issues + element_issues

        # KPI dimensionnels (simple)
        completeness_ok = 1 if len(blocking_issues) == 0 else 0
        # format ok = pas d'issues "FORMAT:" (dans element_issues)
        format_ok = 1 if not any("FORMAT:" in x for x in element_issues) else 0
        # cohérence ok = pas d'issues "COHERENCE:" (bloquantes ou non)
        coherence_ok = 1 if not any("COHERENCE:" in x for x in all_issues) else 0

        ok_completeness += completeness_ok
        ok_format += format_ok
        ok_coherence += coherence_ok

        rows_details.append({
            "IfcClass": ifc_class,
            "GlobalId": gid,
            "SGP_DI_CODE_BIM": code_bim,
            "LocalisationMode": loc_mode,
            "IssuesCount": len(all_issues),
            "BlockingIssuesCount": len(blocking_issues),
            "CompletenessOK": completeness_ok,
            "FormatOK": format_ok,
            "CoherenceOK": coherence_ok,
        })

        rows_codes.append({"GlobalId": gid, "IfcClass": ifc_class, "SGP_DI_CODE_BIM": code_bim})

        for issue in all_issues:
            rows_issues.append({
                "IfcClass": ifc_class,
                "GlobalId": gid,
                "SGP_DI_CODE_BIM": code_bim,
                "Issue": issue,
                "Severity": "BLOCKING" if issue in blocking_issues else "INFO",
            })

    details_df = pd.DataFrame(rows_details)
    issues_df = pd.DataFrame(rows_issues)
    codes_df = pd.DataFrame(rows_codes)

    # 7) Unicité globale CODE_BIM (bloquant)
    duplicates_df = pd.DataFrame()
    if not codes_df.empty:
        tmp = codes_df.dropna(subset=["SGP_DI_CODE_BIM"])
        dups = tmp[tmp.duplicated(subset=["SGP_DI_CODE_BIM"], keep=False)]
        if not dups.empty:
            duplicates_df = dups.sort_values(["SGP_DI_CODE_BIM", "GlobalId"])
            # inject issues
            for _, r in duplicates_df.iterrows():
                issues_df = pd.concat([issues_df, pd.DataFrame([{
                    "IfcClass": r["IfcClass"],
                    "GlobalId": r["GlobalId"],
                    "SGP_DI_CODE_BIM": r["SGP_DI_CODE_BIM"],
                    "Issue": "UNICITE: CODE_BIM en doublon",
                    "Severity": "BLOCKING",
                }])], ignore_index=True)

            # mettre à jour details_df BlockingIssuesCount
            if not details_df.empty:
                dup_ids = set(duplicates_df["GlobalId"].tolist())
                details_df.loc[details_df["GlobalId"].isin(dup_ids), "BlockingIssuesCount"] += 1
                details_df.loc[details_df["GlobalId"].isin(dup_ids), "IssuesCount"] += 1

    # KPI globaux (taux)
    kpi_global = {}
    if total == 0:
        kpi_global = {
            "ElementsAnalyzed": 0,
            "CompletenessRate": 0.0,
            "FormatRate": 0.0,
            "CoherenceRate": 0.0,
            "UniquenessRate": 0.0,
            "GlobalScore": 0.0,
            "BlockingIssues": 0,
            "TotalIssues": 0,
        }
    else:
        completeness_rate = ok_completeness / total
        format_rate = ok_format / total
        coherence_rate = ok_coherence / total

        # unicité rate = 1 si pas de doublons, sinon proportion non-dupliquée
        if duplicates_df.empty:
            uniqueness_rate = 1.0
        else:
            dup_ids = set(duplicates_df["GlobalId"].tolist())
            uniqueness_rate = (total - len(dup_ids)) / total

        # Score global lisible (moyenne des 4)
        global_score = (completeness_rate + format_rate + coherence_rate + uniqueness_rate) / 4

        kpi_global = {
            "ElementsAnalyzed": total,
            "CompletenessRate": completeness_rate,
            "FormatRate": format_rate,
            "CoherenceRate": coherence_rate,
            "UniquenessRate": uniqueness_rate,
            "GlobalScore": global_score,
            "BlockingIssues": int((issues_df["Severity"] == "BLOCKING").sum()) if not issues_df.empty else 0,
            "TotalIssues": len(issues_df) if not issues_df.empty else 0,
        }

    # KPI par classe IFC
    summary_df = pd.DataFrame()
    if not details_df.empty:
        grp = details_df.groupby("IfcClass", as_index=False).agg(
            Elements=("GlobalId", "count"),
            AvgIssues=("IssuesCount", "mean"),
            AvgBlockingIssues=("BlockingIssuesCount", "mean"),
            CompletenessRate=("CompletenessOK", "mean"),
            FormatRate=("FormatOK", "mean"),
            CoherenceRate=("CoherenceOK", "mean"),
        )
        grp = grp.sort_values(["CompletenessRate", "Elements"], ascending=[True, False])
        summary_df = grp

    # Trier détails
    if not details_df.empty:
        details_df = details_df.sort_values(["BlockingIssuesCount", "IssuesCount"], ascending=False)

    # Trier issues
    if not issues_df.empty:
        issues_df = issues_df.sort_values(["Severity", "IfcClass"], ascending=[True, True])

    return summary_df, details_df, issues_df, kpi_global


def to_excel_bytes(summary_df: pd.DataFrame, details_df: pd.DataFrame, issues_df: pd.DataFrame, kpi_global: Dict[str, Any]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([kpi_global]).to_excel(writer, sheet_name="KPI_GLOBAL", index=False)
        summary_df.to_excel(writer, sheet_name="KPI_PAR_CLASSE", index=False)
        details_df.to_excel(writer, sheet_name="DETAILS_OBJETS", index=False)
        issues_df.to_excel(writer, sheet_name="ANOMALIES", index=False)
    return output.getvalue()


def pct(x: float) -> str:
    return f"{x*100:.1f}%"


# =========================
# UI STREAMLIT
# =========================
st.set_page_config(page_title="SGP BIM Value Checker", layout="wide")
st.title("SGP BIM Value Checker — démonstrateur (upload IFC → KPI SGP)")

st.markdown(
    """
Ce démonstrateur permet d’**objectiver la qualité de la donnée BIM** (codification / GDC / patrimoine) à partir d’une maquette **IFC** uploadée.

Il contrôle notamment :
- **Complétude** des Pset SGP (IDENTIFIANT / LOCALISATION / UNICITE / REFERENCES / CODE_BIM)
- **Format** (MAJUSCULES, pas d’espace, pas d’accent, séparateur `-`)
- **Cohérence** (concat IDENTIFIANT, concat LOCALISATION, et `CODE_BIM = LOCALISATION-UNICITE-IDENTIFIANT`)
- **Unicité** (`CODE_BIM` non dupliqué)

> Prototype de thèse : outil d’objectivation / démonstration. Il peut être industrialisé ensuite.
"""
)

uploaded = st.file_uploader("Dépose ton fichier IFC ici", type=["ifc"])

if uploaded is None:
    st.info("Upload un fichier IFC pour lancer l’analyse.")
    st.stop()

with st.spinner("Analyse IFC en cours… (selon la taille, ça peut prendre quelques dizaines de secondes)"):
    temp_path = save_upload_to_temp(uploaded)
    model = ifcopenshell.open(temp_path)
    summary_df, details_df, issues_df, kpi_global = analyze_ifc(model)

# Bandeau KPI globaux
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Score global", pct(kpi_global["GlobalScore"]))
c2.metric("Complétude", pct(kpi_global["CompletenessRate"]))
c3.metric("Format", pct(kpi_global["FormatRate"]))
c4.metric("Cohérence", pct(kpi_global["CoherenceRate"]))
c5.metric("Unicité", pct(kpi_global["UniquenessRate"]))

st.divider()

# Interprétation feu tricolore (simple)
score = kpi_global["GlobalScore"]
if score >= 0.90:
    st.success("🟢 Niveau très satisfaisant : la donnée est globalement exploitable (selon ces règles).")
elif score >= 0.75:
    st.warning("🟠 Niveau intermédiaire : la donnée est partiellement exploitable, corrections nécessaires.")
else:
    st.error("🔴 Niveau insuffisant : la donnée présente des risques importants pour un usage patrimonial.")

# Tables
left, right = st.columns([1.1, 1])

with left:
    st.subheader("KPI par classe IFC")
    if summary_df.empty:
        st.write("Aucune classe analysable (IFC vide ou sans IfcProduct).")
    else:
        st.dataframe(summary_df, use_container_width=True)

with right:
    st.subheader("Objets les plus problématiques (top 50)")
    if details_df.empty:
        st.write("Aucun objet analysé.")
    else:
        st.dataframe(details_df.head(50), use_container_width=True)

st.subheader("Anomalies (top 300)")
if issues_df.empty:
    st.success("Aucune anomalie détectée selon les règles actuelles.")
else:
    st.dataframe(issues_df.head(300), use_container_width=True)
    st.caption("Affichage limité pour la lisibilité. Le rapport Excel contient l’ensemble des anomalies.")

# Téléchargement rapport
excel_bytes = to_excel_bytes(summary_df, details_df, issues_df, kpi_global)
st.download_button(
    "Télécharger le rapport Excel",
    data=excel_bytes,
    file_name="SGP_BIM_Value_Checker_Rapport.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

with st.expander("Notes techniques (pour la thèse)"):
    st.markdown(
        """
- Le périmètre d’analyse est **IfcProduct** (donc “toutes classes” métiers du modèle).
- Les Pset SGP sont lus via `ifcopenshell.util.element.get_psets()` (si disponible).
- Les contrôles sont volontairement **lisibles** : ils produisent des KPI compréhensibles par des non-BIM.
- Ce démonstrateur peut être enrichi (IDS, listes de valeurs, criticité par domaine, etc.).
"""
    )
