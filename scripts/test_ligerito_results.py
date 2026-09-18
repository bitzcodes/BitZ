import copy
import unittest
from ligerito_results import VERSION, validate_ligerito, validate_result_fields


def report(johnson=True):
    """Small metadata fixture; cryptographic validation lives in Rust tests."""
    return dict(protocol_version=VERSION, requested_profile="custom:3:4" if johnson else "udrg:3:4",
                resolved_profile="custom:3:4" if johnson else "udrg:3:4", configuration_fingerprint="a"*64,
                regime="johnson" if johnson else "udr", target_bits=100, outer_ood=johnson,
                outer_ood_grinding_bits=0 if johnson else None, outer_ood_raw_bits=105 if johnson else None,
                recursive_ood=[0, 1 if johnson else 0],
                configuration=dict(hash="blake3", target_security_bits=100,
                    levels=[dict(regime="johnson_ood" if johnson else "udr", ood_samples=n,
                                 log_inv_rate=3+i)
                            for i,n in enumerate([0, 1 if johnson else 0])]))


class LigeritoResultsTests(unittest.TestCase):
    def test_both_regimes_and_actual_caption(self):
        for johnson in [True, False]:
            r=report(johnson)
            validate_ligerito(r,100)

    def test_missing_historical_and_conflicting_metadata_rejected(self):
        for key,value in [("protocol_version","old"),("configuration_fingerprint",None),("outer_ood",False),
                          ("target_bits",106),("recursive_ood",[0,0]),("requested_profile","")]:
            r=report();r[key]=value
            with self.assertRaises(ValueError):validate_ligerito(r,100)
        with self.assertRaises(ValueError):validate_ligerito(None)

    def test_versioned_result_encoding_and_historical_rejection(self):
        import json
        identity=json.dumps(report()).encode().hex()
        self.assertEqual(validate_result_fields(dict(schema="bitz/2",ligerito_hex=identity)),report())
        for fields in [dict(schema="bitz/1",ligerito_hex=identity),dict(schema="bitz/2"),dict(schema="bitz/2",ligerito_hex="xyz")]:
            with self.assertRaises(ValueError):validate_result_fields(fields)

if __name__ == "__main__": unittest.main()
