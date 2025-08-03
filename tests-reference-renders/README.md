# Regression Testing System

This directory contains reference renders used for regression testing the raytracer. The regression testing system ensures that changes to the codebase don't introduce visual regressions in the rendered output.

## How It Works

The regression testing system:

1. Renders test scenes using the current codebase
2. Compares the output against reference renders stored in this directory
3. Uses SSIM (Structural Similarity Index) to detect visual differences
4. Fails if any renders differ significantly from the references

## Running Tests

To run the regression tests locally:

```bash
python3 regression_test.py test
```

This will:

- Render all test scenes
- Compare against reference renders
- Report any differences

## Updating References

To update the reference renders (e.g. after intentional changes):

```bash
python3 regression_test.py render
```

## CI Integration

The regression tests run automatically on every pull request via GitHub Actions. Any visual regressions will cause the CI build to fail.

## Configuration

The regression testing system supports:

- Different render presets (test vs showcase)
- Custom SSIM thresholds
- Image and animation comparisons
- Showcase scene handling

See `regression_test.py` for full configuration options.

## Best Practices

- Always run regression tests before merging changes
- Update reference renders when making intentional visual changes
- Add new test scenes for significant new features
- Keep reference renders up-to-date with the latest codebase
