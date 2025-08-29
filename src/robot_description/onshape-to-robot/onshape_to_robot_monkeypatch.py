from onshape_to_robot import assembly as otor_assembly

# Keep a reference to the original method
_original_find_relations = otor_assembly.Assembly.find_relations

def patched_find_relations(self, *args, **kwargs):
    try:
        # Call the original, but wrap the problematic part
        for relation in getattr(self, "relations", []):
            queries = relation.get("queries", [])
            if len(queries) > 1 and "message" in queries[1]:
                fid = queries[1]["message"].get("featureId")
                if not fid:
                    print("Skipping relation with missing featureId")
                    continue
                feat = self.get_feature_by_id(fid)
                if not feat or "message" not in feat:
                    print(f"Skipping relation — target feature {fid} not found")
                    continue
        # After filtering, call original
        return _original_find_relations(self, *args, **kwargs)
    except TypeError as e:
        print(f"Patched find_relations caught error: {e}")
        return None

# Apply monkey patch
otor_assembly.Assembly.find_relations = patched_find_relations