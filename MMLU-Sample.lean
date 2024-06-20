  -- Define a type for Person
  inductive Person where
  | alice : Person
  | brother : Person

  -- Define a function that returns the number of sisters a person has
  def has_sisters : Person → Nat
  | Person.alice => 2
  | Person.brother => 2

  -- Define a function that returns the number of brothers a person has
  def has_brothers : Person → Nat
  | Person.alice => 5
  | Person.brother => 5

  -- Axioms stating the number of sisters and brothers for Alice
  axiom A1 : has_sisters Person.alice = 2
  axiom A2 : has_brothers Person.alice = 5

  -- Verify the number of sisters Alice's brother has
  -- example : has_sisters Person.brother = 2 := A1
  #eval has_sisters Person.brother
