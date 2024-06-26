-- Define a type for Person
inductive Person where
| rapunzel : Person
| sister1 : Person
| sister2 : Person
| sister3 : Person
| brother1 : Person
| brother2 : Person
deriving DecidableEq

-- Define a function that returns the number of sisters a person has
def has_sisters : Person → Nat
| Person.rapunzel => 3
| Person.sister1 => 3
| Person.sister2 => 3
| Person.sister3 => 3
| Person.brother1 => 4  -- Including Rapunzel
| Person.brother2 => 4  -- Including Rapunzel

-- Define a function that returns the number of brothers a person has
def has_brothers : Person → Nat
| Person.rapunzel => 2
| Person.sister1 => 2
| Person.sister2 => 2
| Person.sister3 => 2
| Person.brother1 => 1
| Person.brother2 => 1

-- Axioms stating the number of sisters and brothers for Rapunzel
axiom A1 : has_sisters Person.rapunzel = 3
axiom A2 : has_brothers Person.rapunzel = 2

-- Verify the number of sisters Rapunzel's brother has (including Rapunzel)
#eval has_sisters Person.brother1
