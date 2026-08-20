# Usage: awk -f scripts/mesh_connected_components.awk mesh.face

function root(x, r, parent_x) {
  if (!(x in parent)) {
    parent[x] = x
  }
  r = x
  while (parent[r] != r) {
    r = parent[r]
  }
  while (parent[x] != x) {
    parent_x = parent[x]
    parent[x] = r
    x = parent_x
  }
  return r
}

function join(a, b, root_a, root_b) {
  root_a = root(a)
  root_b = root(b)
  if (root_a != root_b) {
    parent[root_b] = root_a
  }
}

NR > 3 {
  join($1, $2)
  join($2, $3)
  join($3, $1)
  if ($1 > num_vertices) num_vertices = $1
  if ($2 > num_vertices) num_vertices = $2
  if ($3 > num_vertices) num_vertices = $3
}

END {
  for (i = 1; i <= num_vertices; i++) {
    sizes[root(i)]++
  }
  for (component_root in sizes) {
    num_components++
    if (sizes[component_root] > largest_component) {
      largest_component = sizes[component_root]
    }
  }
  printf("vertices=%d components=%d largest_vertices=%d largest_fraction=%.6f\n",
         num_vertices, num_components, largest_component,
         largest_component / num_vertices)
}
