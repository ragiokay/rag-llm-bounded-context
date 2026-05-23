
Identification requirements from Bounded Context

Continuing the list of specific definitions and examples.
LLM Identifies Business Logic from use cases
Configure LLM for Causal Relation identification
Configure LLM for hierarchy identification
LLM identifies Domain Events
LLM identifies Commands
LLM identifies Actors of Commands
LLM identifies Commands / Events pairs
LLM identifies Policies
LLM groups Command/Events into Aggregates
LLM groups Aggregates into Bounded Contexts

We describe each need to illustrate the details.

107


1. LLM Identifes Business Logic from use cases


What is Business Logic?
Business Logic is the set of rules and operations of a company that contribute towards their goals. They define how real business decisions must be reflected in the systems and software that is used in the company.
Example of business logic in sentences:
In: 
“VIP customers get a 10% discount if their order exceeds $200.”
“When an invoice is overdue by 30 days, automatically add a 5% late fee and suspend the account.”
Out: 
VIP Customers, discounts, discount conditions.
Invoices, late fees, account suspension.

97


2. Configure LLM for Causal Relation identification

What is Causal Relation identification?
It’s basically identifying cause-effect link between elements in text (in our case between words in a sentence). Simply put a “who does what?” or “what causes what?”
Example in sentences:
In: 
“The match was postponed because of a thunderstorm.”
“The user clicks " INITIATE_MEETING " button . The initiated meeting is scheduled and the system sends new meeting messages to all participants .”
“The user selects the meeting the user wants to solve conflict . The system sends a request to the backend to solve the conflict .”
Out: 
Cause: thunderstorm, Effect: match was postponed
Event: "meeting message sent". Command: "initiate meeting"
Event: "meeting date changed". Command:  "solve conflict"

108


3. Configure LLM for hierarchy identification

What is Hierarchy Identification?
Hierarchy identification is the process of detecting a whole–part or container–content relationship between two terms in a sentence. It determines which term (the parent) encompasses, owns, or gives rise to the other (the child). The parent term is the broader entity; the child term is something that belongs to, is part of, or is produced by the parent.
Example in sentences:
In: 
The company’s revenue growth exceeded expectations.
The system updates the new initiator’s permissions .
Out: 
Parent: Company, Child: Revenue growth
Parent: initiator, Child: permissions

110

4. LLM identifies Domain Events 

What is?
In Event Storming, Participants (domain experts, developers) read user-stories, requirements, or spoken narratives and pick out all meaningful business occurrences. They write them as past-tense verb phrases (e.g., Order Placed). These events are things that happen in the domain and are important to the business. They represent state changes that domain experts care about. A system “reading” sentences would scan for verbs describing business/system actions and convert them into discrete, time‑ordered events. Not necessarily ordered or a sequence, for our system they can just be events.
Example in sentences:
In: 
The User enters the user's username and password and presses the " LOGIN " button . The user is logged in to the system .
A customer places an order for a lamp. The warehouse checks availability. If it’s in stock, the order is confirmed, payment is taken, and the lamp is shipped.
Out: 
user logged in, user verified.
Order Placed, Inventory Checked, Order Confirmed, Payment Taken, Order Shipped.

112

5. LLM identifies Commands

What is?
In Event Storming, a Command represents an intention to do something in the system. It’s a request or decision made by a user, an external system, or even a timer, expressed as an imperative verb phrase like “Place Order”, “Register User”, or “Approve Invoice”. Commands are the “what the system is asked to do”. They often trigger one or more domain events.
Example in sentences:
In: 
A customer wants to add a new shipping address after they log in.
The system receives the participant ’s attending form . The system sets the user ’s participate status to " YES " .
Out: 
Add Shipping Address
Set status

114

6. LLM identifies Actors of Commands

What is?
Actors of Commands are the people, roles, (or external systems) that initiate an action (a command). They express an intent, and that intent becomes a command in the system. The actor is who or what wants something to happen.
Example in sentences:
In: 
A customer wants to add a new shipping address after they log in.
The system sets the user ’s participate status to " YES " .
Out: 
Actor: Customer. Command: Add Shipping Address
Actor: System. Command: Set participate status

116

7. LLM identifies Commands / Events pairs

What is?
Identifying Command-Event pairs means finding these cause‑and‑effect links: when a specific Command is executed, a specific Event occurs. To differentiate from the “Causal Relation identification” slide, in this step we want to use that capability to relate “Commands” with “Events” from the use cases the system receives.
So the slide 2. Is about the capability, trained with general “cause-relation relationships”, this slide/step is for the focus on identifying it in systems.
Examples: (See 2. Causal Relation slide)

118

8. LLM identifies Policies 

What is?
In Event Storming, a Policy is a business rule that continuously listens for a domain event and, when that event occurs, automatically triggers a command to carry out some action. It captures the “When X happens, then do Y” logic of a system.
In OUR system, the event-command pairs are often considered Policies. Because they explain how system actions occur, this is how Events trigger Commands or viceversa.
Example in sentences:
In: 
When a payment is overdue, block the account.
The invited participant submits the attending form. The system decides the participant ’s attending time .
Out: 
WHEN PaymentOverdue THEN BlockAccount
"form submitted"->"receive form"

128


9. LLM groups Command/Events into Aggregates

What is?
Aggregate is a consistency boundary that clusters one or more domain objects (entities & value objects). It enforces business invariants (rules that must always hold) and acts as the single entry point for changes, this is receiving commands and emitting events.
In a few words, it’s a group of highly related elements, around a specific concept, and one change could change all of them, as they are a whole.
Example in sentences:
In: 
“When a customer submits an order, the total must be greater than zero and each line item must reference a valid product.”
User, user profile, user level, user setting, user status.            (Entity, attribute, Value objects)
Out: 
Order Aggregate
User Aggregate

130

10. LLM groups Aggregates into Bounded Contexts

What is?
A Bounded Context is a clear boundary around a part of a system where a specific domain model and its own language (ubiquitous language) are valid. In Event Storming, you discover these boundaries by grouping events, commands, and aggregates that naturally belong together and speak the same language, separating them from other parts that use different meanings for the same terms.
While aggregates group elements that are close together and make a single concept, Bounded Contexts group those concepts into a bigger scale to make a working module of a system.
Examples:
In: 
"A customer browses products and adds a few to the cart. When they proceed to checkout, a payment is authorized. After payment, the warehouse picks the items, packs them, and ships the order."
Meeting Aggregate, Participant Aggregate, Meeting initiator Aggregate. User Aggregate, User Handling Aggregate
Out: 
Shopping Context, Payment Context, Fulfillment Context.
Meeting Scheduling Context. User Management Context.

132
