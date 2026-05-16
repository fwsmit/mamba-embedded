import onnx
from onnx import shape_inference, helper, numpy_helper
import numpy as np

def fix_transpose_rank_mismatch(model_path, output_path):
    model = onnx.load(model_path)
    inferred = shape_inference.infer_shapes(model)
    graph = inferred.graph

    # Build concrete shape map for every tensor
    shape_map = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        s = vi.type.tensor_type.shape
        if s:
            shape_map[vi.name] = [d.dim_value for d in s.dim]

    new_nodes = []
    fix_idx = 0

    for node in graph.node:
        if node.op_type == "Transpose":
            inp = node.input[0]
            in_shape = shape_map.get(inp, [])
            perm = next(
                (list(a.ints) for a in node.attribute if a.name == "perm"), None
            )

            if perm and len(in_shape) > 0 and len(in_shape) != len(perm):
                target_rank = len(perm)
                print(f"Mismatch: in_shape={in_shape}, perm={perm}")

                # Merge all middle dims: [B, d1, d2, ..., L] -> [B, d1*d2*..., L]
                # Preserves batch (dim 0) and sequence (last dim), collapses everything between
                merged_middle = int(np.prod(in_shape[1:-1]))
                new_shape = [in_shape[0], merged_middle, in_shape[-1]]
                assert len(new_shape) == target_rank, \
                    f"Cannot reconcile shape {in_shape} to rank {target_rank}"

                shape_name = f"__fix_shape_{fix_idx}__"
                out_name   = f"__fix_reshape_{fix_idx}__"
                fix_idx += 1

                graph.initializer.append(
                    numpy_helper.from_array(
                        np.array(new_shape, dtype=np.int64), name=shape_name
                    )
                )
                new_nodes.append(
                    helper.make_node("Reshape",
                                     inputs=[inp, shape_name],
                                     outputs=[out_name])
                )
                # Update shape map so any further nodes in same pass see correct rank
                shape_map[out_name] = new_shape
                node.input[0] = out_name
                print(f"  Inserted Reshape: {in_shape} -> {new_shape}")

        new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)

    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"Saved: {output_path}")

fix_transpose_rank_mismatch("src/models/har-mamba-1.onnx", "src/models/har-mamba-1-fixed.onnx")
