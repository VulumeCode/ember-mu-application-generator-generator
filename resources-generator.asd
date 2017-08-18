(asdf:defsystem :resources-generator
  :name "resources-generator"
  :author "Vincent Goossens <vincent.goossens@tenforce.com"
  :version "0.0.1"
  :maintainer "Vincent Goossens <vincent.goossens@tenforce.com"
  :licence "MIT"
  :description "Generates ember generate commands and dispatcher rules from a domain.lisp file."
  :serial t
  :depends-on (mu-cl-resources)
  :components ((:file "packages")
               (:file "resources-generator")))
