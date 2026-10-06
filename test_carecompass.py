import pytest
import pandas as pd


def test_pkg_resources_importable():
    """Verify that pkg_resources is provided by setuptools<81 without raising ModuleNotFoundError."""
    import pkg_resources  # noqa: F401


def test_llama_index_legacy_import():
    """Verify that Document can be imported from llama_index.legacy."""
    from llama_index.legacy import Document
    assert Document is not None


def test_document_instantiation():
    """Verify that Document can be instantiated with text content."""
    from llama_index.legacy import Document
    doc = Document(text="Test clinical notes for patient")
    assert doc.text == "Test clinical notes for patient"


def dataframe_to_documents(df):
    from llama_index.legacy import Document
    return [
        Document(
            text=f"Index: {int(row['PATIENT_ID'])}\nName: {row['FIRST']}\nDOB: {row['BIRTHDATE']}\nNotes: {row['CLINICAL_NOTES']}"
        )
        for _, row in df.iterrows()
    ]


def test_dataframe_to_documents():
    """Verify that dataframe_to_documents converts a Pandas DataFrame into LlamaIndex Documents."""
    sample_df = pd.DataFrame([
        {
            "PATIENT_ID": 101,
            "FIRST": "Jane",
            "BIRTHDATE": "1985-05-15",
            "CLINICAL_NOTES": "Patient presented with severe fatigue.",
        }
    ])
    documents = dataframe_to_documents(sample_df)
    assert len(documents) == 1
    assert "Index: 101" in documents[0].text
    assert "Name: Jane" in documents[0].text
    assert "Notes: Patient presented with severe fatigue." in documents[0].text


def test_environment_reproducibility():
    """Verify that setuptools is installed and has a version compatible with pkg_resources (<81)."""
    import setuptools
    from packaging.version import parse as parse_version
    version = parse_version(setuptools.__version__)
    assert version < parse_version("81.0.0"), f"setuptools version {setuptools.__version__} is >= 81.0.0"
