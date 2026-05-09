from scripts import compare_task1_results


def test_compare_task1_results_script_writes_report(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(
        compare_task1_results,
        "collect_train_results",
        lambda roots: [
            {
                "payload": {
                    "experiment_id": "exp",
                    "checkpoint_path": str(tmp_path / "model.pt"),
                    "status": "completed",
                    "metrics": {"dev_rollout_metrics": {"score_total_proxy": 1.0}},
                },
                "source_path": str(tmp_path / "train_result.json"),
                "record_type": "train_result",
            }
        ],
    )

    exit_code = compare_task1_results.main(["--output", str(tmp_path / "comparison.json")])

    assert exit_code == 0
    assert (tmp_path / "comparison.json").is_file()
    assert "records: 1" in capsys.readouterr().out
