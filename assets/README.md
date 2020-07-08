# VSRL Assets

Assets are primarily images used for rendering visual environments.

VSRL downloads assets from the public GitHub repository into `~/.vsrl`.
If you would like to modify your local assets, you can do so by first running
an environment and then editing files in your `~/.vsrl` directory.

Project maintainers can update the assets by running:

```
zip -r assets.zip *;
git commit -m "Updated assets." assets.zip;
git push
```

in this directory.
