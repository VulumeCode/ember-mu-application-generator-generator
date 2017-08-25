# Resources generator

Generates `ember generate mu-resource ...` commands and dispatcher rules from a domain.lisp file by introspecting the resulting model.

## Example usage

```
    dr run -v [folder containing domain.lisp]:/config semtech/mu-cl-resources-ember-generator
```

Before exiting, the program outputs `ember generate mu-resource ...` commands and dispatcher rules. The commands have to be run in the root of an Ember project. See the [mu-resource blueprint documentation](https://git.tenforce.com/mu-semtech/ember-mu-application-generator) for more information and options.

The rules can be copied and pasted in a dispatcher.ex file.

## Readonly option

The `ember generate mu-resource ...` command takes a `--readonly` flag. To generate commands with this flag set, run the generator with the environment variable `READONLY` set to any value.

```
    dr run -v [folder containing domain.lisp]:/config semtech/mu-cl-resources-ember-generator --env READONLY=t
```
