# Python Coding and Documentation Standards

Apply these standards to all new and modified Python code and documentation in
this repository. Before completing a change, verify that every added or modified
public API satisfies them. Preserve existing user-written explanations and type
specificity when updating APIs.

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
