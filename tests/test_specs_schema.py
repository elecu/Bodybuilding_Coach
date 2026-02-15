from __future__ import annotations

import unittest

from bbcoach.federations.specs import load_federation_specs


class FederationSpecsSchemaTests(unittest.TestCase):
    def test_required_federations_present(self) -> None:
        lib = load_federation_specs()
        for required in ("ukbff", "bnbf", "wnbf_uk", "pca", "nabba"):
            self.assertIn(required, lib.federations, f"Missing federation spec: {required}")

    def test_division_minimum_fields(self) -> None:
        lib = load_federation_specs()
        self.assertGreaterEqual(len(lib.list_federations()), 5)

        for fed in lib.list_federations():
            self.assertTrue(fed.federation_id)
            self.assertTrue(fed.federation_name)
            self.assertTrue(fed.sources.source_files)
            self.assertTrue(fed.divisions, f"{fed.federation_id} has no divisions")

            for div in fed.divisions:
                self.assertTrue(div.division_id)
                self.assertTrue(div.division_name)
                self.assertTrue(div.sources.source_files, f"{fed.federation_id}/{div.division_id} missing source_files")
                self.assertIsNotNone(div.needs_verification)
                if div.eligibility.age_rule_text:
                    self.assertIsNotNone(
                        div.eligibility.age_rule_semantics,
                        f"{fed.federation_id}/{div.division_id} has age_rule_text without age_rule_semantics",
                    )
                    if (
                        div.eligibility.age_rule_semantics is not None
                        and div.eligibility.age_rule_semantics.needs_verification
                    ):
                        self.assertTrue(
                            div.needs_verification,
                            f"{fed.federation_id}/{div.division_id} unclear age semantics must set needs_verification=true",
                        )

                if div.mandatory_poses is None:
                    self.assertTrue(
                        div.needs_verification,
                        f"{fed.federation_id}/{div.division_id} has null mandatory_poses without needs_verification=true",
                    )
                else:
                    for pose in div.mandatory_poses:
                        text = pose.lower()
                        self.assertNotIn("8 mandatory poses", text)
                        self.assertNotEqual(text.strip(), "mandatory poses")

    def test_selected_refs_can_be_generated(self) -> None:
        lib = load_federation_specs()
        refs = lib.iter_division_refs()
        self.assertTrue(refs, "No selectable division refs generated")
        keys = {ref.key() for ref in refs}
        self.assertEqual(len(keys), len(refs), "Duplicate SelectedDivisionRef keys generated")

    def test_pca_required_divisions_present(self) -> None:
        lib = load_federation_specs()
        pca = lib.get_federation("pca")
        self.assertIsNotNone(pca, "Missing PCA federation spec")
        assert pca is not None
        required = {
            "mens_bodybuilding_junior",
            "mens_bodybuilding_first_timers",
            "mens_bodybuilding_novice",
            "mens_bodybuilding_masters_over_40",
            "mens_bodybuilding_masters_over_50",
            "mens_bodybuilding_mr_classes",
            "classic_bodybuilding",
            "mens_physique_open",
            "mens_physique_junior",
            "mens_physique_masters_35_plus",
            "mixed_disability",
            "ladies_bikini_short_medium_tall",
            "ladies_junior_bikini",
            "ladies_bikini_trained",
            "ladies_bikini_masters_35_plus",
            "ladies_bikini_masters_45_plus",
            "ladies_wellness",
            "ladies_toned_figure",
            "ladies_athletic_figure",
            "ladies_trained_figure",
        }
        division_ids = {div.division_id for div in pca.divisions}
        for division_id in required:
            self.assertIn(division_id, division_ids, f"Missing PCA division: {division_id}")


if __name__ == "__main__":
    unittest.main()
