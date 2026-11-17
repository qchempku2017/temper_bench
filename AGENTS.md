# Python Coding and Documentation Standards

Apply these standards to all new and modified Python code and documentation in
this repository. Before completing a change, verify that every added or modified
public API satisfies them. Preserve existing user-written explanations and type
specificity when updating APIs.

## Design priorities: minimalism and clarity

- Prefer the smallest implementation that fully satisfies the current
  requirement. Every abstraction, intermediate representation, helper, copy,
  and persisted file should have a concrete purpose for an existing caller.
- Use the domain's typed schemas directly. Avoid converting a manifest into
  intermediate configuration dictionaries and reconstructing it afterward when
  the manifest can already supply the required information.
- Avoid convenience wrappers that merely chain one or two straightforward
  calls without adding a useful API or distinct behavior. For example, reading
  a manifest and then loading its referenced bundle does not need a separate
  function that only returns those two objects.
- Do not add speculative recovery paths, reference archives, compatibility
  layers, fallback formats, or duplicate metadata to cover hypothetical needs.
  Preserve validation required for correctness, such as checksums, path
  containment, identities, array shapes, units, and frame alignment.
- Prefer direct, coherent refactors. Update callers and remove superseded code
  when a replacement is authorized. Add compatibility aliases or migration
  machinery only for an established requirement, not automatically.

## Ownership and separation of responsibilities

- Place behavior with the component responsible for its purpose. Merely
  reading a schema object does not make an operation a method of that schema.
  Read-only methods are still appropriate when they naturally describe or
  access that object's own data.
- The pipeline runner constructs execution and result manifests. For example,
  make_test_manifests belongs beside create_result_manifest in the runner;
  MLFFTrainBundle supplies its inputs rather than constructing result records.
- Test manifests provide execution instructions before evaluation, with result
  fields unfilled. Execution fills measured results and writes output files;
  the experiment manifest retains the corresponding test manifests. Avoid a
  second test-configuration format that repeats the same information.
- Job management submits, monitors, downloads, verifies, and cleans up jobs.
  It must not manufacture execution results or rewrite downloaded manifests
  to invent missing information. Keep runner-written results.json and its
  referenced bundle.json unchanged during collection.
- Result loading reads and validates recorded metadata, original labels, and
  predictions, then assembles scientific data. Keep it free of submission,
  execution, reference-archiving, and manifest-reconstruction responsibilities.

## Dataset and result lifecycle

- Users are responsible for retaining original split datasets unchanged at
  their recorded local locations. State this requirement in documentation;
  do not create additional permanent reference copies to compensate for it.
- Copy datasets into submission folders only as needed for execution. Remove
  temporary copies at the appropriate upload or execution cleanup stage.
  Recover reference labels from the original paths recorded in the training
  unit, not from a second content-addressed dataset archive.
- Preserve original test filenames and relative subdirectories through
  packaging, manifests, prediction mappings, and analysis. Avoid replacing
  meaningful names with identifiers such as test_001 when the originals suffice.
- Distinguish original local dataset paths from execution copies. For example,
  execution_input_path identifies a copied input for either local or remote
  execution; it is not the original reference-file location.
- Missing required runner output is a collection failure. In particular, do
  not synthesize a pending manifest when results.json was not downloaded, and
  do not proceed to destructive cleanup after unsuccessful verification.
  Missing or changed original labels must produce a clear analysis error.

## Construction, validation, and explicit refresh

- Establish required derived metadata during ordinary construction or
  validation, before finalizing identity. Avoid requiring callers to construct
  an incomplete object and then create another copy just to complete it.
- Normal TrainingUnit construction calculates missing dataset hashes and
  verifies supplied hashes before finalizing training_unit_id. Supplied hashes
  must not be silently replaced when file contents differ.
- Preserve an explicit metadata-only loading mode for remote execution and
  inspection without original local files. verify_files=False preserves recorded
  hashes and identities without reading datasets; it does not establish that
  the files are suitable for execution.
- Deliberately accepting changed input bytes is an explicit refresh operation.
  A function that returns a refreshed copy must say so in its name and document
  identity recalculation and preservation of the original object. Ordinary
  construction should not require that refresh operation.
- Reuse expensive file hashes within one clearly bounded operation when files
  are unchanged. Do not treat file size or modification time as proof of equal
  bytes, or retain a stale cache across independent operations.

## Shared utilities and dependency direction

- Put shared production helpers in temper.utils, grouped by relevance and functionality:
  file_paths, file_hashing, file_writing, extxyz, dataset_sizes, cuda, and
  defaults. Extend an appropriate existing module before adding another one.
- Check actual callers before placing a helper in the production package.
  Helpers used exclusively by tests belong under tests/, even when several
  test modules share them. Do not expose them through temper.utils for
  hypothetical future reuse.
- Keep path validation separate from file hashing and writing, and extxyz
  metadata parsing in its own module. Avoid miscellaneous utils
  modules scattered throughout schemas and pipelines.
- Keep job-specific operations, such as choosing downloaded outputs, locating
  a job's bundle, or allocating attempt directories, in job_management. A
  function is not a general utility merely because several callers use it.
- Shared utilities must not depend on schemas, pipelines, result loading, or
  job management. Those components import utilities. Keep optional model and
  GPU dependencies lazy so importing a utility does not require those runtimes.
- Import helpers from their defining modules. Do not add blanket re-exports in
  temper.utils.__init__ or enumerate every public function in __all__ without
  a specific need for export control.

## Public and private functions

- Functions deliberately shared across modules should have public names.
  Do not prefix them with an underscore while relying on other modules to
  import them as part of normal operation.
- Keep implementation helpers private when they are only used within their
  owning module. Frequency of use alone does not determine visibility.
- Give private functions explanatory NumPy-style docstrings too. Describe their
  purpose, parameters, return values, path rules, checks, and important failure
  conditions as applicable. An underscore is not a reason to omit documentation.

## Type annotations

- Follow modern Python type annotation conventions: PEP 484, PEP 526, PEP 585,
  and PEP 604.
- Provide explicit type annotations for all function and method parameters and
  return values.
- Use built-in generic types (`list`, `dict`, `tuple`, and `set`) instead of
  deprecated `typing` aliases.
- Use PEP 604 union syntax (`X | Y` and `X | None`) instead of `Union[X, Y]`
  and `Optional[X]`.
- Annotate class and instance attributes wherever practical.
- Avoid `Any` unless genuinely necessary.
- Preserve existing type annotations; do not weaken type specificity.

## Docstring conventions

- Strictly follow NumPy-style docstrings as defined by the
  [numpydoc Style Guide](https://numpydoc.readthedocs.io/en/latest/format.html).
- Give every public class, function, and method a complete, informative docstring.
- Begin each docstring with a concise summary of its purpose.
- Document every parameter in a `Parameters` section, including its type and
  semantic meaning.
- Document returned values and their types in a `Returns` section when applicable.
- Document important exceptions in a `Raises` section, explaining the conditions
  under which they occur.
- Document class initialization parameters in the class-level `Parameters`
  section.
- Document public class and instance attributes in the class-level `Attributes`
  section, including types and descriptions.
- Include `Yields`, `Notes`, `Examples`, or `See Also` sections where appropriate.
- Follow NumPy section formatting and ordering exactly, including section
  headings, dashed underlines, and indented descriptions. For example:

  ```text
  Parameters
  ----------
  output_dir : pathlib.Path
      Directory in which to write results. Relative paths are resolved against
      the current working directory.
  ```

- Do not use Google-style (`Args:`) or Sphinx field-list-style (`:param:`)
  docstrings.
- Do not omit documentation merely because information is already present in
  type annotations.
- Keep docstrings synchronized with implementation changes.

## Documentation quality

- Explain the semantics and constraints of parameters and attributes, not merely
  their names or types.
- Explicitly describe path resolution, units, default behavior, accepted values,
  side effects, and failure conditions wherever relevant.
- Explain when a value is absent, what that absence means for the caller, and
  what operations are valid in that state.
- Ensure that documentation generators can extract complete API information
  from the docstrings.

### Preserve caller-oriented explanations

Write docstrings for the person using the object. Explain what a value contains,
where its paths are resolved, when it is absent, and what that absence means for
the caller. Describe return values and side effects concretely. Avoid compressed
implementation jargon. Preserve user-written explanations when updating APIs.

For example, `MLFFTrainBundle.files` is absent on a newly constructed local bundle;
packaging fills it with copied input paths and output locations, and the runtime
cannot train or evaluate an unpackaged bundle.

## Documentation clarity and pedagogical style

- Write for scientifically and technically literate readers, not exclusively
  for professional software engineers. Do not assume familiarity with
  specialized software engineering terminology, architectural patterns, or
  industry jargon.
- Prefer direct, precise, pedagogical explanations over jargon-heavy, abstract,
  or unnecessarily sophisticated language.
- Do not sacrifice technical accuracy or necessary technical depth for
  simplicity. Explain important technical concepts rather than avoiding them.
- When documenting a function, class, feature, or concept, explain:
  1. What it is or does: its purpose and observable behavior.
  2. What it affects: its inputs, outputs, side effects, constraints, and
     consequences.
  3. How to use it: how callers or users should interact with it, with examples
     where helpful.
  4. Why it exists, when relevant: the problem it solves or the rationale behind
     a non-obvious design decision.
- Explain behavior and semantics rather than merely restating code structure,
  method names, or implementation details.
- Avoid unexplained jargon, buzzwords, vague abstractions, and fashionable
  software engineering expressions. Define necessary specialized terms in plain
  language on first use.
- Do not substitute metaphors or informal language for precise technical
  explanations.
- Prefer concrete descriptions of actual behavior over abstract descriptions of
  responsibilities, workflows, or architecture.
- Describe error conditions, path resolution rules, assumptions, and limitations
  explicitly whenever they affect correct usage.
- Avoid circular explanations such as "the manager manages resources" or
  "the adapter adapts the backend."
- Keep simple concepts concise. Add detail only when it helps readers understand
  or correctly use the API.
- Documentation should teach readers how to understand and use the software,
  not merely demonstrate familiarity with software engineering terminology.

## Naming conventions

- Use PascalCase for all class names, following PEP 8. Examples:
  `BaseMLFFAdapter`, `MLFFTrainBundle`, and `TrainingConfig`.
- Use snake_case for all function and method names, following PEP 8. Examples:
  `run_test`, `load_training_config`, and `resolve_bundle_paths`.
- Choose names that are self-explanatory, semantically precise, and as concise
  as reasonably possible.
- Communicate the purpose or behavior of a class, function, or method without
  requiring readers to inspect its implementation.
- Avoid excessively generic names such as `Manager`, `Handler`, `Processor`,
  `Helper`, `Utils`, `process_data`, or `handle_request` unless their meaning is
  genuinely clear from context.
- Avoid unnecessarily long or redundant names. Do not repeat information already
  obvious from the enclosing class, module, or namespace.
- Prefer meaningful words over obscure abbreviations. Preserve widely understood
  scientific and domain-specific abbreviations such as `DFT`, `MLFF`, and `MD`.
- Prefer verb-based names for functions and methods that perform actions, and
  noun-based names for classes representing objects, configurations, or concepts.
- For boolean-returning functions, prefer predicate names such as `is_valid`,
  `has_labels`, or `can_train`, where appropriate.
- Maintain consistent terminology across related classes, functions,
  configuration fields, and documentation.
- Prefer explicit relationships over compressed noun combinations: for example,
  resolve_path_in_bundle, load_bundle_from_manifest, and
  load_results_from_manifest. Add locality when it resolves a real ambiguity,
  as in resolve_local_dataset_path.
- Make copying and mutation semantics clear in names. For example,
  copy_with_refreshed_dataset_hashes returns a new object; a plain refresh name
  could misleadingly suggest an in-place operation.
- Name files after what they actually contain. Use manifest_io for manifest
  access and download_utils for download-related operations; avoid vague or
  obsolete terms such as report or collection when they obscure the role.
- Do not rename existing public APIs solely for stylistic consistency without
  considering backward compatibility and updating their references.

## Review before completing a change

- Check annotations on added or modified parameters, return values, and
  attributes for completeness and specificity.
- Check that all added or modified public APIs have complete NumPy-style
  docstrings with the applicable sections and correct formatting.
- Verify that documented types, defaults, accepted values, path resolution,
  units, return values, side effects, and failure conditions match the code.
- Confirm that explanations are useful to callers, preserve existing meaningful
  documentation, and define any necessary specialized terminology.
- Check naming consistency and backward compatibility when changing public APIs.
- Follow renames and moves through imports, callers, tests, docstrings, examples,
  and documentation. Search for stale references and remove unused imports and
  superseded files. Do not leave two implementations of the same responsibility.
- Keep existing behavioral coverage when refactoring. Run relevant tests, and
  run the full suite for broad changes to schemas or shared utilities. Verify
  dependency and optional-import boundaries when moving shared code.
- Report the concrete changes and the checks actually run, including failures
  or skipped tests. Do not claim that a rename or passing unit tests establishes
  successful execution on a real remote host or model backend.
