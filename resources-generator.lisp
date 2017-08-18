(in-package :resources-generator)

; Generate a string with a list of eg commands
(defun generate-resources ()
  (format t "~{~A~%~}"
          (mapcar #'generate-resource (all-resources))))

; Generate a string with a list of dispatcher rules
(defun generate-dispatchers ()
  (format t "~{~A~^ ~%~}"
          (mapcar #'generate-dispatcher (all-resources))))


; Fetch all resources from domain.lisp
(defun all-resources ()
  (loop for val being
     the hash-values of mu-cl-resources::*resources*
     collect val))




; Generate a string with an eg command for one resource
(defun generate-resource (resource)
  "Returns the generator for the specific resource"
  (format nil "~A ~A ~{~A ~}~:{~A:belongs-to:~A~#[~;~~~A~] ~}~:{~A:has-many:~A~#[~;~~~A~] ~}"
          "ember g mu-resource" ; terminal command
          (gen-resource-name resource)  ; resource name
          (mapcar #'gen-resource-slot (mu-cl-resources::ld-properties resource)) ; its attributes
          (mapcar #'gen-resource-rel (mu-cl-resources::has-one-links  resource)) ; its belongs-to relations
          (mapcar #'gen-resource-rel (mu-cl-resources::has-many-links resource)) ; its has-many relations
          ))

(defun gen-resource-name (resource)
  (string-downcase (mu-cl-resources::resource-name resource)))

(defun gen-resource-slot (property)
  (format nil "~A:~A"
          (string-downcase (symbol-name (mu-cl-resources::json-key property)))
          (string-downcase (symbol-name (mu-cl-resources::resource-type property)))))

(defun gen-resource-rel (link)
  (let ((i (find-inverse link)))
        (if i
            `(,(mu-cl-resources::request-path link) ,(string-downcase (mu-cl-resources::resource-name link)) ,i)
            `(,(mu-cl-resources::request-path link) ,(string-downcase (mu-cl-resources::resource-name link)))
        )
  )
)

(defun find-inverse (link)
  (let ((i (mu-cl-resources::inverse-links link)))
        (if i
            (string-downcase (mu-cl-resources::request-path (getf (car i) :link)))
        )
  )
)



; Generate a string with a dispatcher rule for one resource
(defun generate-dispatcher (resource)
  "Returns the dispatcher rule for the specific resource"
  (let ((path (string-downcase (mu-cl-resources::request-path resource))))
      (format nil "  match \"/~A/*path\" do~%    Proxy.forward conn, path, \"http://resource/~A/\"~%  end"
          path
          path
      )
  )
)
