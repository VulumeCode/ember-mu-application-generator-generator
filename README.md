# Resources generator

Generates `ember generate mu-resource ...` commands and dispatcher rules from a domain.lisp file by introspecting the resulting model.

## Example usage

```
    dr run -v [folder containing domain.lisp]:/config semtech/mu-cl-resources-ember-generator
```

Before exiting, the program outputs `ember generate mu-resource ...` commands and dispatcher rules. The commands have to be run in the root of an Ember project see the mu-resource blueprint documentation for more information and options.

The rules can be copied and pasted in a dispatcher.ex file.
