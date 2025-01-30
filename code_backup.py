# %%
bed_types = ['Double/Full Bed', 'Futon Bed', 'King Bed', 'Murphy Bed', 'Queen Bed', 'Sofa Bed', 'Twin/Single Bed', 'Single Bed', 'Run of the house', 
             'Dorm/Bunk Bed', 'Water Bed']
room_types = ['Accessible Room', 'Suite', 'Executive/Club Suite', 'Double Room', 'King Bedroom', 'Apartment', 'Queen Bedroom', 'Penthouse', 
              'Studio Suite', 'Twin Room', 'Family Room/Suite', 'Cabin', 'Cottage', 'Loft', 'Guest Room', 'Bungalow', 'Villa', 'Junior Suite']

# %%
rate_plan_inclusives_raw = """
Piano
24-hour front desk
24-hour room service
24-hour security
adjoining rooms
Air conditioning
Airline desk
ATM/Cash machine
Baby sitting
BBQ/Picnic area
Bilingual staff
Bookstore
Boutiques/stores
Brailed elevators
Business library
Care rental desk
Casino
Check cashing policy
Check-in kiosk
Cocktail lounge
Coffee Shop
Coin operated laundry
Concierge desk
Concierge floor
Conference facilities
Courtyard
Currency exchange
Desk with electrical outlet
Doctor on call
Door man
Driving Range
Drugstore/pharmacy
Duty free shop
Elevators
Exective floor
Exercise gym
Express Check-In
Express Check-out
Family plan
Florist
Folios
free shuttle
free parking
Free Transportation
Game room
Gift/News Stand
Hairdresser/Barber
Accessible Facilities
Health Club
Heated Pool
Housekeeping - Daily
Housekeeping - Weekly
Ice Machine
Indoor Parking
Indoor Pool
Hot Tub
Jogging Track
Kennels
Laundry/Valet Service
Liquor Store
Live Entertainment
Massage Services
Nightclub
Off-site parking
On-site parking 
Outdoor parking
Outdoor Pool
Package/Parcel services
Parking
Photocopy Center
Playground
Pool
Poolside Snack bar
Public Address systemRamp Access
Ramp Access
Recreational Vehicle Parking
Restaurant
Room Service 
Safe deposit box
Sauna
Security
Shoe Shine Standa
Shopping Mall
Solarium
Spa
Sports Bar
Steam Bath
Storage Space
Sundry/Convenience Store
Technical Concierge 
Theatre Desk
Tour/sightseeing desk
Translation services
Travel agency
Truck Parking 
Valet cleaning
Dry cleaning
Valet Parking 
Vending Machine
Video Tapes
Wakeup services
Wheelchair access
Whirlpool
Multilingual staff
Wedding services
Banquet facilities
Bell staff/porter
Beauty shop/salon
Complimentary self service laundry
Direct  dial telephone 
Female traveler room/floor
Pharmacy
Stables
120 AC
120 DC
220 AC
Accessible Parking
220 DC
Barbeque Grills
Women's clothing
Men's clothing
Children's clothing
Shops and commercial services
Video games
Sports bar open for lunch
Sports bar open for dinner
Room Service - full menu
Room Service - Limited menu
Room Service - limited hours
Valet same day dry clening
Body scrub
Body wrap
Public area air conditioned
Efolio available to company
Induvidual Efolio available
Video review billinh
All-incusive meal plan
Meal plan availabe
Modified american meal plan
Food and Beverage outlets
Lounges/bars
Barber shop
Video checkout
Onsite laundry
24-hour food & beverage kiosk
Concierge lounge
Parking fee managed by Hotel
Transportation
Breakfast served in restaurant
Lunch served in restaurant
Dinner served in restaurant
Full service housekeeping
Limited service housekeeping
High speed internet access for laptop in public area
Wirless internet connection in public area
Additional services/amenities/facilities on property
Transportation services - local area
Transportation srevices - local office
DVD/video rental
Parking lot
Parking deck
Street side parking
Cocktail lounge with entertainment
Cocktail lounge with light fare
Motorcycle parking 
Phone services
Ballroom
Bus parking
Children's play area
Children's nursery
Disco
Early Check-in
Locker room
Non-smoking rooms (generic)
Train access
Aerobics instruction
Baggage hold
Bicycle rentals
Dietician
Late check-out available
Pet-sitting services
prayer mats
Sports trainer
Turndown service
DVSs/videos - children
Bank
Lobby coffee service
Banking services
Stairwells
Pet amenities available
Exhibition/convention floor
Long term parking
Children not alowed
Children welcome
Courtesy car
Hotel does not provide porographic films/TV
Hotspots
Free high speed internet connections
Internet services
Pets allowed
Gourmet highlights
Catering services
Complimentary breakfast
Business center
Business services 
Secured parking
Racquetball
Snow sports
Tennis court
Water sports
Child programs
Golf
Horseback riding
Oceanfront
Beachfront
Hair dryer
Ironing board
Heated guest rooms
Toilet
Parlor
Video game player
Thalassotheraphy
Private dining for groups
Hearing impaired services
Carryout breakfast
Deluxe Continental brekfast
Hot continental breakfast
Hot breakfast
Private pool
Connecting rooms
Data port
Exterior Corridors
Gulf view
ADA accessible
High speed internet access  
Interior corridors
High speed wireless
Kitchenette
Private bath or shower
Fire safety compliant
Welcome Drink
Boarding pass print-out available
Printing services available
All public areas non-smoking
Meeting rooms
Movies in room
Secretarial service
Snow skiing 
Water skiing 
Fax service
Great room 
Lobby coffee service
Multiple phone lines billed separetly
Umbrellas
Gas station
Grocery store
24-hour coffee shop
Airport shuttle service
Luggage service
Piano bar
VIP security
Complimetary wirless internet
 Concierge breakfast
Same gender floor
Children programs
Buildin meets local, state and countru building codes
Internet browser on TV
Newspaper
Parking - controlled access gates to enter parking area
Hotel safe deposit box (not room safe box)
Storage space available - fee
Type of entrances to guest rooms
Beverage/cocktail
Cell phone rental
Coffee/tea
Early check in guarantee
Food and beverage discount
Late check out guarantee
Room upgrade confirmed
Room upgrade on availability
Shuttle to local businesses
Shuttle to local attractions
Social hour
Video billing
Welcome gift 
Hypoallergenic rooms
Room air filtration
Smoke-free property
Water purification system in use
Poolside service
Clothing store
Electric car charging stations
Office rental
Piano
Incoming fax
Outgoing fax
Semi-private space
Loading dock
Baby kit
Children's breakfast
Cloakroom service
Coffee lounge
Events ticket service
Late check-in
Limited parking 
Outdoor summer bar/caf?
No parking available
Beer garden
Garden lounge bar
Summer terrace
Winter terrace
Roof terrace
Beach bar
Helicopter service
Ferry
Tapas bar
Caf? Bar
Snack bar
Guestroom wired internet
Guestroom wireless internet
Fietness center
Alcoholic beverages
Non-alcoholic beverages
Health and beauty services
Eco Friendly 
Stay safe
Rooms with balcony
Local calls
Minibar
Refrigerator
In Room Safe
Smoking rooms available
Free WIFI in meeting rooms
Beach view
Ocean view
Mountain view
Pool view
Family Room
Rollaway adult
Crib charge
Extra person
Disney Park Tikcets included
Water Park Passes included
Free Minibar
Ski pass included
Free Airport Parking
Free Ski Lift Ticket & Rental
Free one way Airport transfer
Massage services included
Spa Access included 
Golf access included
Free valet parking
Train station shuttle services
Electric car charging stations
Golf clubs (equipments)
Beach club
Theme park shuttle
"""
rate_plan_inclusives = [line.strip() for line in rate_plan_inclusives_raw.split("\n") if line.strip()]
len(rate_plan_inclusives)


# %%
room_amenities_raw = """
Adjoining rooms
Air conditioning
Alarm clock
All news channel
radio
Baby listening device
balcony
Barbeque grills
Bathtub with spray jets
Bathrobe
Bathroom amenities
Bathroom telephone
Bathtub
Bathtub only
Bathtub & shower
Bidet
Bottled water
Cable television
Coffee maker
Color television
Computer
Connecting rooms
electric converters 
Copier
Cordless phone
Cribs
Data port
Desk
Desk with lamp
Dining guide
Direct dial phone number
Dishwasher
Double beds
Dual voltage outlet
Electrical current voltage
Ergonomic chair
Extended phone cord
Fax machine
Fire alarm
Fire alarm with light
Fireplace
Free toll free calls
Free calls
Free credit card access calls
Free local calls
Free movies
Full kitchen
Grab bars in bathroom
Grecian tub
Hairdryer
High speed internet connection
Interactive web TV
International direct dialing
Internet access
Iron
Ironing board
Whirlpool
King bed
Kitchen
Kitchen supplies
Kitchenette
Knock light
Laptop
Large desk
Large work area
Laundry basket
Loft
Microwave
Minibar
Modem
Modem jack
Multi-line phone
Newspaper
Non-smoking
Notepads
Office supplies
Oven
Pay per view movies on TV
Pens
Phone in bathroom
Plates and bowls
Pots and pans
Prayer mats
Printer
Private bathroom
Queen bed
Recliner
Refrigerator
Refrigerator with ice maker
Remote control television
Rollaway bed
Safe
Scanner
Separate closet
Separate modem line available
Shoe polisher
Shower only
Silverware
Sitting area
Smoke detectors
Smoking
Sofa bed
Speaker phone
Stereo
Stove
Tape recorder
Telephone
Telephone for hearing impaired
Telephones with message light
Toaster oven
Trouser press
Turn down service
Twin bed
Vaulted ceilings
VCR movies
VCR player
Video games
Voice mail
Wake-up calls
Water closet
Water purification system
Wet bar
Wireless internet connection
Wireless keyboard
Adaptor available for telephone PC use
Air conditioning individually controlled in room
Bathtub & whirlpool
Telephone with data ports
CD player
Complimentary local calls time limit
Extra person charge for rollaway use
feather pillows
Desk with electrical outlet
ESPN available
Foam pillows
HBO available
Ceiling fan
DVD player
Mini refrigerator
Separate line billing for multi-line phone
Self-controlled heating/cooling system
Toaster
Analog data port
Collect calls
International calls
Carrier access
Interstate calls
Intrastate calls
Local calls
Long distance calls
Operator assisted calls
Credit card access calls
Calling card calls
Toll free calls
Universal adaptors
Bathtub seat
Canopy
glassware
Entertainment center
Family room
Hypoallergenic bed
Hypoallergenic pillows
Lamp
Meal included - breakfast
Meal included - continental breakfast
Meal included - dinner
Meal included - lunch
Shared bathroom
Textphone
Water bed
Extra adult charge
Extra child charge
Extra child charge for rollaway use
Meal included: full American breakfast
Futon
Murphy bed
Tatami mats
Single bed
Annex room
Free newspaper
Honeymoon suites
Complimentary high speed internet in room
Maid service
PC hook-up in room
Satellite television
VIP rooms
Cell phone recharger
DVR player
iPod docking station
Media center
Plug & play panel
Satellite radio
Video on demand
Exterior corridors
Gulf view
Accessible room
Interior corridors
Mountain view
Ocean view
High speed internet access for fee
High speed wireless
Premium movie channels
Slippers
First nighters' kit
Chair provided with desk
Pillow top mattress
Feather bed
Duvet
Luxury linen type
International channels
Pantry
Dish-cleaning supplies
Double vanity
Lighted makeup mirror
Upgraded bathroom amenities
VCR player available at front desk
Instant hot water
Outdoor space
Hinoki tub
Private pool
High Definition (HD) Flat Panel Television - 32 inches or greater
Room windows open
Bedding type unknown or unspecified
Full bed
Round bed
TV
Child rollaway
DVD player available at front desk
Video game player:
Video game player available at front desk
Dining room seats
Full size mirror
Mobile/cellular phones
Movies
Multiple closets
Plates/glassware
Safe large enough to accommodate a laptop
Bed linen thread count
Blackout curtain
Bluray player
Device with mp3
No adult channels or adult channel lock
Non-allergenic room
Pillow type
Seating area with sofa/chair
Separate toilet area
Web enabled
Widescreen TV
Other data connection
Phoneline billed separately
Separate tub or shower
Video games
Roof ventilator
Children's playpen
Plunge pool
DVD movies
Air filtration
Exercise Equipment in Room
Towels Sheets
Electric Kettle
Portable Fan
Washing Machine
Towels Provided
Smart TV
Streaming Services
Highchair
Toilet with electronic bidet
Free tea bags instant coffee
Outdoor Shower
Baby bath
Changing table
Heating/Heated floor
Indoor private mineral hot springs (japanese style onsen)
Freezer
Ice maker
In room childcare
Digital TV serviceWired Internet Access
Wired Internet Access
Pillow Menu
Room Size(Square feet)
Room Size (Square meters)
Smart TV
Cleaning Supplies
Dryer
Tatami (Woven mat) floors
Private spa tub
Cookware, dishware, and utensils
"""
room_amenities = [line.strip() for line in room_amenities_raw.split("\n") if line.strip()]
len(room_amenities)


# %%
meal_plan_raw = """
All Inclusive
Full Board or American
Buffet Breakfast
Continental Breakfast
Family Plan
Full Board
Half Board or Modified American
Room Only or European
Self Catering
Breakfast and Dinner
Breakfast 
Lunch
Dinner
Breakfast and Lunch
Lunch and Dinner
"""
meal_plan = [line.strip() for line in meal_plan_raw.split("\n") if line.strip()]
len(meal_plan)

# %%
room_view_raw = """
Airport view
Bay view
City view
Courtyard view
Golf View
Harbor view
Intercoastal view
Lake view
Marina view
Mountain view
Ocean view
Pool view
River view
Water view
Beach view
Garden view
Park view
Forest view
Rain forest view
Various views
Limited view
Slope view
Strip view
Countryside view
Sea view
Gulf view
Landmark View
Resort View
Lagoon View
Street View
Panoramic View
Hill View
Valley View
Vineyard View
Desert View
Mansion View
Metro View
Balcony View
Rock View
Skyline View
Village View
Stadium View
Island View
Runway View
Avenue View
Tower View
Fountain View
Cathedral View
Church View
Sunset View
"""
room_view = [line.strip() for line in room_view_raw.split("\n") if line.strip()]
len(room_view)

# %%
taxes_fees_raw = """
Bed tax
City hotel fee
PropertyFee
City tax
Municipal Tax
County tax
Energy tax
Federal tax
Food & beverage tax
Lodging tax
Maintenance fee
Occupancy tax
HotelOccupancyTax
Package fee
Resort fee
ResortFee
Sales tax
Taxes
SalesTax
Service charge
ServiceFee
State tax
Province Tax
Surcharge
Total tax
Tourism tax
VAT/GST tax
Surplus Lines Tax
Insurance Premium Tax
Application Fee
Express Handling Fee
Exempt
Standard
Zero-rated
Miscellaneous
MandatoryTax
Room Tax
Tax
Early checkout fee
Country tax
Extra person charge
ExtraPersonFee
Banquet service fee
Room service fee
Local fee
Goods and services tax (GST)
Value Added Tax (VAT)
Crib fee
Rollaway fee
Assessment/license tax
Pet sanitation fee
Not known
Child rollaway charge
Convention tax
Extra child charge
Standard food and beverage gratuity
National government tax
Adult rollaway fee
Beverage with alcohol
Beverage without alcohol
Tobacco
Food
Total surcharges
TaxAndServiceFee
State cost recovery fee
Miscellaneous fee
Mandatory Fee
MandatoryFee
Destination amenity fee
Refundable pet fee
Property Fee
Tax Recovery Charges and Service Fees
Tax Recovery Charges
"""
taxes_fees = [line.strip() for line in taxes_fees_raw.split("\n") if line.strip()]
len(taxes_fees)