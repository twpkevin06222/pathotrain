# pathotrain

Training on whole slide images (WSI), built on
[patchology](https://github.com/twpkevin06222/patchology) for tiling.

> [!WARNING]
> For research use only. Not a medical device.

## Logging

Logs (via loguru) show `INFO` and above by default. Each library has its own
level, so you can turn one up without the other:

```bash
export PATHOTRAIN_LOG_LEVEL=DEBUG    # pathotrain: DEBUG, ALL for everything, WARNING for less
export PATCHOLOGY_LOG_LEVEL=WARNING  # patchology (tiling) messages
```

If your application configures loguru itself (removes the default handler
before importing pathotrain), that setup is left untouched.

## License

Apache License 2.0, see [LICENSE](LICENSE).
