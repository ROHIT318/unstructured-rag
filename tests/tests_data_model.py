# Tests for utils/data_model.py.

import importlib
import sys
from types import SimpleNamespace

import pytest


class FakeUploadedFile:
    """Mimics streamlit's UploadedFile: has a .name and .getbuffer()."""

    def __init__(self, name, content=b"file-bytes"):
        self.name = name
        self._content = content

    def getbuffer(self):
        return self._content


class FakeForm:
    def __init__(self, key):
        self.key = key

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeStatus:
    def __init__(self, state, label, expanded):
        self.state = state
        state["status_labels"].append(label)
        state["status_updates"] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, **kwargs):
        self.state["status_updates"].append(kwargs)


def make_fake_streamlit(state):
    def text_input(label, *args, **kwargs):
        state["text_inputs"].append(label)
        return state.get("text_input_values", {}).get(label, "")

    return SimpleNamespace(
        form=lambda key: FakeForm(key),
        file_uploader=lambda *a, **k: state["uploaded_files"],
        text_input=text_input,
        form_submit_button=lambda *a, **k: state["submitted"],
        write=lambda msg: state["writes"].append(msg),
        status=lambda label, expanded=False: FakeStatus(state, label, expanded),
        spinner=lambda *a, **k: FakeStatus(state, *a, **k) if a else None,
    )


@pytest.fixture
def state():
    return {
        "uploaded_files": [],
        "submitted": False,
        "writes": [],
        "text_inputs": [],
        "status_labels": [],
        "status_updates": [],
    }


def run_form(monkeypatch, tmp_path, state, store_mocks):
    """Run data_model.py once with the scripted state and return the captured state.

    store_mocks: dict of function name -> MagicMock replacing the pipeline
    functions that data_model imports.
    """
    # The module writes uploads to relative paths under data/, so run it from
    # a temp directory to avoid touching the real repo folders.
    (tmp_path / "data" / "images").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "vector_db").mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)

    # Import data_pipeline for real (its st.write calls hit the fake too), then
    # patch the store_* functions so no embedding/Chroma work happens.
    import utils.data_pipeline as dp
    for name, mock in store_mocks.items():
        monkeypatch.setattr(dp, name, mock)

    # Swap in the fake streamlit and force a fresh import of data_model.
    monkeypatch.setitem(sys.modules, "streamlit", make_fake_streamlit(state))
    monkeypatch.delitem(sys.modules, "utils.data_model", raising=False)
    importlib.import_module("utils.data_model")

    return state


@pytest.fixture
def store_mocks():
    from unittest.mock import MagicMock
    return {
        "store_pdf": MagicMock(return_value="Storing PDF operation completed....."),
        "store_image": MagicMock(return_value=("collection", ["id"])),
        "store_ppt": MagicMock(return_value="Storing PPT operation completed....."),
    }


class TestFormRendering:

    def test_no_submission_processes_nothing(self, monkeypatch, tmp_path, state, store_mocks):
        state["uploaded_files"] = [FakeUploadedFile("a.pdf")]
        state["submitted"] = False

        state = run_form(monkeypatch, tmp_path, state, store_mocks)

        store_mocks["store_pdf"].assert_not_called()
        assert state["status_labels"] == []
        assert list((tmp_path / "data" / "vector_db").iterdir()) == []

    def test_submission_with_no_files_completes(self, monkeypatch, tmp_path, state, store_mocks):
        state["submitted"] = True

        state = run_form(monkeypatch, tmp_path, state, store_mocks)

        for mock in store_mocks.values():
            mock.assert_not_called()
        # Status was opened and marked complete.
        assert state["status_updates"] and state["status_updates"][-1]["state"] == "complete"

    def test_collects_model_name_and_description(self, monkeypatch, tmp_path, state, store_mocks):
        state["submitted"] = False
        state = run_form(monkeypatch, tmp_path, state, store_mocks)
        assert state["text_inputs"] == ["Model Name", "model_description"]


class TestPdfDispatch:

    def test_pdf_saved_to_temp_and_removed_after(self, monkeypatch, tmp_path, state, store_mocks):
        state["submitted"] = True
        state["uploaded_files"] = [FakeUploadedFile("report.pdf", b"pdf-content")]
        state["text_input_values"] = {"Model Name": "my_model"}

        run_form(monkeypatch, tmp_path, state, store_mocks)

        store_mocks["store_pdf"].assert_called_once()
        call = store_mocks["store_pdf"].call_args
        temp_path = call.kwargs["path"]
        assert call.kwargs["model_name"] == "my_model"
        # Temp file had the expected prefix/suffix but has been cleaned up.
        assert temp_path.startswith("data/vector_db/temp_")
        assert temp_path.endswith("_report.pdf")
        assert not (tmp_path / temp_path).exists()

    def test_pdf_bytes_written_before_store(self, monkeypatch, tmp_path, state, store_mocks):
        state["submitted"] = True
        state["uploaded_files"] = [FakeUploadedFile("report.pdf", b"pdf-content")]

        captured = {}

        def fake_store_pdf(**kwargs):
            captured["existed"] = False
            with open(kwargs["path"], "rb") as f:
                captured["content"] = f.read()
            return "done"

        store_mocks["store_pdf"].side_effect = fake_store_pdf
        run_form(monkeypatch, tmp_path, state, store_mocks)
        assert captured["content"] == b"pdf-content"

    def test_empty_model_name_still_forwarded(self, monkeypatch, tmp_path, state, store_mocks):
        state["submitted"] = True
        state["uploaded_files"] = [FakeUploadedFile("report.pdf")]

        run_form(monkeypatch, tmp_path, state, store_mocks)
        assert store_mocks["store_pdf"].call_args.kwargs["model_name"] == ""


class TestImageDispatch:

    @pytest.mark.parametrize("extension", ["jpeg", "jpg", "png"])
    def test_image_saved_permanently_and_kept(self, monkeypatch, tmp_path, state, store_mocks, extension):
        state["submitted"] = True
        state["uploaded_files"] = [FakeUploadedFile(f"photo.{extension}", b"img-bytes")]
        state["text_input_values"] = {"Model Name": "img_model"}

        run_form(monkeypatch, tmp_path, state, store_mocks)

        call = store_mocks["store_image"].call_args
        saved_path = call.kwargs["path"]
        assert saved_path.startswith("data/images/")
        assert saved_path.endswith(f"_photo.{extension}")
        assert call.kwargs["model_name"] == "img_model"
        # Unlike pdf/pptx, the image file is kept on disk after storing.
        assert (tmp_path / saved_path).exists()

    def test_multiple_images_each_dispatched(self, monkeypatch, tmp_path, state, store_mocks):
        state["submitted"] = True
        state["uploaded_files"] = [FakeUploadedFile("a.png"), FakeUploadedFile("b.jpg")]

        run_form(monkeypatch, tmp_path, state, store_mocks)
        assert store_mocks["store_image"].call_count == 2


class TestPptDispatch:

    def test_pptx_saved_to_temp_and_removed_after(self, monkeypatch, tmp_path, state, store_mocks):
        state["submitted"] = True
        state["uploaded_files"] = [FakeUploadedFile("deck.pptx", b"pptx-bytes")]
        state["text_input_values"] = {"Model Name": "ppt_model"}

        run_form(monkeypatch, tmp_path, state, store_mocks)

        call = store_mocks["store_ppt"].call_args
        temp_path = call.kwargs["path"]
        assert temp_path.startswith("data/vector_db/temp_")
        assert temp_path.endswith("_deck.pptx")
        assert call.kwargs["model_name"] == "ppt_model"
        assert not (tmp_path / temp_path).exists()


class TestUnsupportedTypes:

    @pytest.mark.parametrize("name", ["data.csv", "data.xlsx", "data.xlsm", "data.xlsb", "file.docx"])
    def test_unhandled_extension_as_only_file_raises_name_error(
            self, monkeypatch, tmp_path, state, store_mocks, name):
        # Known bug: the csv/xlsx/else branches never assign `result`, so
        # st.write(result) raises NameError when such a file comes first.
        state["submitted"] = True
        state["uploaded_files"] = [FakeUploadedFile(name)]

        with pytest.raises(NameError):
            run_form(monkeypatch, tmp_path, state, store_mocks)

        for mock in store_mocks.values():
            mock.assert_not_called()

    def test_csv_after_handled_file_writes_stale_result(
            self, monkeypatch, tmp_path, state, store_mocks):
        # Same bug, other direction: a csv after a pdf re-writes the pdf's result.
        state["submitted"] = True
        state["uploaded_files"] = [FakeUploadedFile("a.pdf"), FakeUploadedFile("b.csv")]

        run_form(monkeypatch, tmp_path, state, store_mocks)

        store_mocks["store_pdf"].assert_called_once()
        # The pdf result was written twice (once for each file).
        assert state["writes"].count("Storing PDF operation completed.....") == 2

    def test_extension_without_dot_matches(self, monkeypatch, tmp_path, state, store_mocks):
        # endswith("pdf") also matches a file literally named "mypdf" — the
        # dispatch tries to open it as a pdf.
        state["submitted"] = True
        state["uploaded_files"] = [FakeUploadedFile("mypdf", b"bytes")]

        run_form(monkeypatch, tmp_path, state, store_mocks)
        store_mocks["store_pdf"].assert_called_once()


class TestMultipleFiles:

    def test_mixed_batch_dispatches_each_correctly(self, monkeypatch, tmp_path, state, store_mocks):
        state["submitted"] = True
        state["uploaded_files"] = [
            FakeUploadedFile("a.pdf"),
            FakeUploadedFile("b.png"),
            FakeUploadedFile("c.pptx"),
        ]
        state["text_input_values"] = {"Model Name": "mixed_model"}

        run_form(monkeypatch, tmp_path, state, store_mocks)

        assert store_mocks["store_pdf"].call_count == 1
        assert store_mocks["store_image"].call_count == 1
        assert store_mocks["store_ppt"].call_count == 1
        for mock in store_mocks.values():
            assert mock.call_args.kwargs["model_name"] == "mixed_model"

    def test_processing_message_written_per_file(self, monkeypatch, tmp_path, state, store_mocks):
        state["submitted"] = True
        state["uploaded_files"] = [FakeUploadedFile("a.pdf"), FakeUploadedFile("b.png")]

        run_form(monkeypatch, tmp_path, state, store_mocks)
        assert "Processing file a.pdf" in state["writes"]
        assert "Processing file b.png" in state["writes"]

    def test_store_results_surfaced_in_ui(self, monkeypatch, tmp_path, state, store_mocks):
        state["submitted"] = True
        state["uploaded_files"] = [FakeUploadedFile("a.pdf")]

        run_form(monkeypatch, tmp_path, state, store_mocks)
        assert "Storing PDF operation completed....." in state["writes"]
