-- Define a type for Person
inductive Person where
  | alice : Person
  | alice_brother1 : Person
  | alice_brother2 : Person
  | alice_brother3 : Person
  | alice_brother4 : Person
  | alice_sister : Person
  deriving DecidableEq

-- Define the number of sisters for Alice and her brothers
def has_sisters : Person → Nat
  | Person.alice => 1
  | Person.alice_brother1 => 2
  | Person.alice_brother2 => 2
  | Person.alice_brother3 => 2
  | Person.alice_brother4 => 2
  | Person.alice_sister => 1
  | _ => 0

-- Define the number of brothers for Alice
def has_brothers : Person → Nat
  | Person.alice => 4
  | Person.alice_sister => 4
  | _ => 0

-- Define the function to calculate the number of sisters for Alice's brothers
def number_of_sisters (p : Person) : Nat :=
  if p = Person.alice_brother1 ∨ p = Person.alice_brother2 ∨ p = Person.alice_brother3 ∨ p = Person.alice_brother4 then
    has_sisters p
  else 0

#eval number_of_sisters Person.alice_brother1 -- Evaluate the number of sisters for Alice's brother
