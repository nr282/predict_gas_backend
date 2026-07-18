"""
The purpose of this module is to standardize location from strings
and also provide mappings between (1) location names and (2) longtitude and lattitude.

For instance, the goal of this module is to be able to take a string such as "tvagWashingtonDC"
must be mapped to "Washington, DC".

In addition, we would like to be able to go from the string "Washington, DC" to a longtitude and lattitude.
Likewise, it will be good to longtitude and lattitude to "Washington, DC".

"""
import logging
from collections import namedtuple
location = namedtuple('Location', ['Latitude', 'Longitude'])

def get_list_of_standardizied_name():

    return ["Washington, DC",
            "Virginia",
            "Accomack County, Virginia",
            "Albemarle County, Virginia",
            "Alleghany County, Virginia",
            "Amelia County, Virginia",
            "Amherst County, Virginia",
            "Appomattox County, Virginia",
            "Arlington County, Virginia",
            "Augusta County, Virginia",
            "Bath County, Virginia",
            "Bedford County, Virginia",
            "Bland County, Virginia",
            "Bennington County, Virginia",
            "Botetourt County, Virginia",
            "Brunswick County, Virginia",
            "Buchanan County, Virginia",
            "Buckingham County, Virginia",
            "Campbell County, Virginia",
            "Caroline County, Virginia",
            "Carroll County, Virginia",
            "Charles City County, Virginia",
            "Charlotte County, Virginia",
            "Chesterfield County, Virginia",
            "Clarke County, Virginia",
            "Craig County, Virginia",
            "Culpeper County, Virginia",
            "Cumberland County, Virginia",
            "Dare County, Virginia",
            "Davis County, Virginia",
            "DeKalb County, Virginia",
            "Delaware County, Virginia",
            "Dickenson County, Virginia",
            "Dinwiddie County, Virginia",
            "Douglas County, Virginia",
            "Edwards County, Virginia",
            "Essex County, Virginia",
            "Fairfax County, Virginia",
            "Fairfax City County, Virginia",
            "Farragut County, Virginia",
            "Fauquier County, Virginia",
            "Floyd County, Virginia",
            "Fluvanna County, Virginia",
            "Franklin County, Virginia",
            "Frederick County, Virginia",
            "Giles County, Virginia",
            "Gloucester County, Virginia",
            "Goochland County, Virginia",
            "Grayson County, Virginia",
            "Greene County, Virginia",
            "Greensville County, Virginia",
            "Hampton County, Virginia",
            "Halifax County, Virginia",
            "Hanover County, Virginia",
            "Henrico County, Virginia",
            "Henry County, Virginia",
            "Highland County, Virginia",
            "Isle of Wight County, Virginia",
            "Jamestown County, Virginia",
            "James City County, Virginia",
            "King and Queen County, Virginia",
            "King George County, Virginia",
            "King William County, Virginia",
            "Lancaster County, Virginia",
            "Lee County, Virginia",
            "Loudoun County, Virginia",
            "Louisa County, Virginia",
            "Lunenburg County, Virginia",
            "Madison County, Virginia",
            "Mathews County, Virginia",
            "Mecklenburg County, Virginia",
            "Middlesex County, Virginia",
            "Monroe County, Virginia",
            "Montgomery County, Virginia",
            "Nelson County, Virginia",
            "New Kent County, Virginia",
            "Northampton County, Virginia",
            "Northumberland County, Virginia",
            "Nottoway County, Virginia",
            "Orange County, Virginia",
            "Page County, Virginia",
            "Patrick County, Virginia",
            "Pittsylvania County, Virginia",
            "Powhatan County, Virginia",
            "Prince Edward County, Virginia",
            "Prince William County, Virginia",
            "Pulaski County, Virginia",
            "Rappahannock County, Virginia",
            "Richmond County, Virginia",
            "Roanoke County, Virginia",
            "Rockbridge County, Virginia",
            "Rockingham County, Virginia",
            "Russell County, Virginia",
            "Scott County, Virginia",
            "Shenandoah County, Virginia",
            "Smyth County, Virginia",
            "Southampton County, Virginia",
            "Spotsylvania County, Virginia",
            "Stafford County, Virginia",
            "Surry County, Virginia",
            "Sussex County, Virginia",
            "Tazewell County, Virginia",
            "Warren County, Virginia",
            "Washington County, Virginia",
            "Westmoreland County, Virginia",
            "Wise County, Virginia",
            "Wythe County, Virginia",
            "York County, Virginia",
            "Alexandria city, Virginia",
            "Bristol city, Virginia",
            "Buena Vista city, Virginia",
            "Charlottesville city, Virginia",
            "Chesapeake city, Virginia",
            "Colonial Heights city, Virginia",
            "Covington city, Virginia",
            "Danville city, Virginia",
            "Emporia city, Virginia",
            "Fairfax city, Virginia",
            "Falls Church city, Virginia",
            "Franklin city, Virginia",
            "Fredericksburg city, Virginia",
            "Galax city, Virginia",
            "Hampton city, Virginia",
            "Harrisonburg city, Virginia",
            "Hopewell city, Virginia",
            "Lexington city, Virginia",
            "Lynchburg city, Virginia",
            "Manassas city, Virginia",
            "Manassas Park city, Virginia",
            "Martinsville city, Virginia",
            "Newport News city, Virginia",
            "Norfolk city, Virginia",
            "Norton city, Virginia",
            "Petersburg city, Virginia",
            "Poquoson city, Virginia",
            "Portsmouth city, Virginia",
            "Prince George County, Virginia",
            "Radford city, Virginia",
            "Richmond city, Virginia",
            "Roanoke city, Virginia",
            "Salem city, Virginia",
            "Staunton city, Virginia",
            "Suffolk city, Virginia",
            "Virginia Beach city, Virginia",
            "Waynesboro city, Virginia",
            "Williamsburg city, Virginia",
            "Winchester city, Virginia"]


#TODO: We need to turn the above into long and latt.


def raw_name_to_standard_name(raw_name):
    """
    Maps the (a) raw name to the (b) standard name.

    """

    logging.debug(f"Raw name is provided by: {raw_name}")

    result = None
    if "Washington" in raw_name and "DC" in raw_name:
        result = "Washington, DC"

    if not (result in get_list_of_standardizied_name()):
        return False

    return result




