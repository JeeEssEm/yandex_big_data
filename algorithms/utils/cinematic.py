from math import cos, radians


def distance(longitude1, latitude1, longitude2, latitude2):
    degree_kilometres = 111.2
    latitude_dif = degree_kilometres * abs(latitude2 - latitude1)
    longitude_dif = abs(longitude1 - longitude2)
    AD = degree_kilometres * cos(radians(latitude1)) * longitude_dif
    BC = degree_kilometres * cos(radians(latitude2)) * longitude_dif
    temp = (AD - BC) / 2
    H = (latitude_dif ** 2 - temp ** 2) ** 0.5
    return ((max(AD, BC) - temp) ** 2 + H ** 2) ** 0.5 * 1000
