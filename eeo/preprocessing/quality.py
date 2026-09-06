"""Decoding the quality layers that ship beside a product's reflectance bands.

Every Sentinel-2 and Landsat product carries a per-pixel quality layer, and it
is the only honest way to know which pixels in a scene are actually a view of
the ground. The layers are not comparable to each other: Sentinel-2's ``SCL``
holds one class number per pixel, where Landsat's ``QA_PIXEL`` packs several
independent flags into the bits of a single integer. This module's job is to
turn either of them into the same thing — a boolean array that is ``True``
where a pixel should be discarded — so that everything downstream can be
written once rather than once per mission.

Deciding *which* pixels count as cloud is a judgement rather than a fact, so
the class sets below are defaults and not truths. They are named constants
precisely so that a user who disagrees can say so in their own code, and so
that a published analysis can state exactly what it masked.

Notes
-----
Class numbers are labels, not quantities: the midpoint of "cloud shadows" (3)
and "vegetation" (4) is not "somewhere between the two", it is "not vegetated"
(5). A quality band must therefore only ever be resampled by nearest
neighbour, which is what :attr:`~eeo.io._bands.BandInfo.kind` marks these
bands for. Their dtype after a load is not significant — stacking an integer
``SCL`` alongside float reflectance bands widens it to float — but the values
themselves must have arrived unmodified.

References
----------
Class numbers and names follow the Scene Classification table published by
Copernicus at https://sentiwiki.copernicus.eu/web/s2-processing.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable
from enum import IntEnum

import numpy as np

from eeo.core.exceptions import ValidationError


class SCLClass(IntEnum):
    """The twelve classes of the Sentinel-2 Level-2A scene classification.

    The ``SCL`` band assigns every pixel exactly one of these numbers. Member
    names are Copernicus's own, spelled as they appear in the published Scene
    Classification table, so that code reads the same as the documentation a
    user is checking it against. The members are ordered by class number,
    which is also their value, so a member can be used anywhere the raw number
    can::

        SCLClass.CLOUD_HIGH_PROBABILITY == 9

    Two classes have been renamed by ESA since the mission launched, without
    their numbers changing: class 2 was ``DARK_FEATURES`` ("dark features /
    shadows") before the 2022 processing-baseline change, and class 5 is
    widely documented under its older informal name "bare soil". Both older
    spellings are still accepted by :func:`resolve_scl_classes`.

    Attributes
    ----------
    NO_DATA : int
        0 — outside the granule, or otherwise without a measurement.
    SATURATED_OR_DEFECTIVE : int
        1 — the sensor saturated, or the detector reported a defect.
    CAST_SHADOWS : int
        2 — topographic cast shadows: ground shadowed by terrain rather than
        by cloud. Named ``DARK_FEATURES`` in older products and documentation.
    CLOUD_SHADOWS : int
        3 — shadow cast by cloud, detected from cloud geometry.
    VEGETATION : int
        4 — vegetated surface.
    NOT_VEGETATED : int
        5 — a surface that is not vegetated. The class is defined negatively
        and so conflates bare soil, rock and built surfaces; older
        documentation calls it "bare soil", which is narrower than what it
        actually contains.
    WATER : int
        6 — water.
    UNCLASSIFIED : int
        7 — no class was confidently assigned. Not a synonym for "clear": it
        marks where the classifier could not decide, so it is neither a claim
        that the pixel is ground nor that it is cloud.
    CLOUD_MEDIUM_PROBABILITY : int
        8 — probably cloud.
    CLOUD_HIGH_PROBABILITY : int
        9 — almost certainly cloud.
    THIN_CIRRUS : int
        10 — thin high-altitude ice cloud. It scatters and attenuates rather
        than hiding the ground, so a pixel here holds a real measurement of a
        real surface — but a contaminated one.
    SNOW_ICE : int
        11 — snow or ice. Bright and cold, and routinely confused with cloud
        in both directions.

    Examples
    --------
    >>> SCLClass.CLOUD_SHADOWS
    <SCLClass.CLOUD_SHADOWS: 3>
    >>> int(SCLClass.THIN_CIRRUS)
    10
    >>> SCLClass(9).name
    'CLOUD_HIGH_PROBABILITY'
    """

    NO_DATA = 0
    SATURATED_OR_DEFECTIVE = 1
    CAST_SHADOWS = 2
    CLOUD_SHADOWS = 3
    VEGETATION = 4
    NOT_VEGETATED = 5
    WATER = 6
    UNCLASSIFIED = 7
    CLOUD_MEDIUM_PROBABILITY = 8
    CLOUD_HIGH_PROBABILITY = 9
    THIN_CIRRUS = 10
    SNOW_ICE = 11


# Spellings ESA has moved away from, plus the singular of a plural class name.
# Accepted because the old names are what most tutorials, papers and existing
# scripts say, and silently failing to resolve "bare_soil" would be a worse
# outcome than accepting it.
_SCL_NAME_ALIASES = {
    "dark_features": SCLClass.CAST_SHADOWS,
    "dark_feature": SCLClass.CAST_SHADOWS,
    "dark_area": SCLClass.CAST_SHADOWS,
    "dark_area_pixels": SCLClass.CAST_SHADOWS,
    "cast_shadow": SCLClass.CAST_SHADOWS,
    "casted_shadows": SCLClass.CAST_SHADOWS,
    "cloud_shadow": SCLClass.CLOUD_SHADOWS,
    "bare_soil": SCLClass.NOT_VEGETATED,
    "bare_soils": SCLClass.NOT_VEGETATED,
    "not_vegetated_soil": SCLClass.NOT_VEGETATED,
    "snow_or_ice": SCLClass.SNOW_ICE,
    "snow": SCLClass.SNOW_ICE,
    "ice": SCLClass.SNOW_ICE,
}

#: The classes Easy-EO treats as cloud by default: cloud shadows (3), both
#: cloud probabilities (8, 9), and thin cirrus (10). Medium probability is
#: included because excluding it keeps a band of half-cloud around every cloud
#: edge, and cirrus because a contaminated measurement is still a wrong one.
#: It is a deliberately cautious set: it will discard some clear ground, which
#: costs pixels, rather than admit cloud, which costs correctness.
SCL_CLOUDY: frozenset[SCLClass] = frozenset(
    {
        SCLClass.CLOUD_SHADOWS,
        SCLClass.CLOUD_MEDIUM_PROBABILITY,
        SCLClass.CLOUD_HIGH_PROBABILITY,
        SCLClass.THIN_CIRRUS,
    }
)

#: The classes that carry no usable measurement at all — no data (0) and
#: saturated or defective (1) — whatever the sky was doing. Masking these is
#: not a cloud judgement and there is no sensible reason to keep them.
SCL_NODATA: frozenset[SCLClass] = frozenset(
    {
        SCLClass.NO_DATA,
        SCLClass.SATURATED_OR_DEFECTIVE,
    }
)

#: What :func:`scl_mask` masks when not told otherwise: :data:`SCL_CLOUDY`
#: together with :data:`SCL_NODATA`.
SCL_DEFAULT_MASKED: frozenset[SCLClass] = SCL_CLOUDY | SCL_NODATA


def _resolve_scl_class(value: SCLClass | int | str) -> SCLClass:
    """Resolve one class number or class name to an :class:`SCLClass`."""
    if isinstance(value, str):
        name = value.strip().upper()
        if name in SCLClass.__members__:
            return SCLClass[name]
        alias = _SCL_NAME_ALIASES.get(name.lower())
        if alias is not None:
            return alias
        raise ValidationError(
            f"no scene class named {value!r}; the names are "
            f"{', '.join(member.name.lower() for member in SCLClass)}"
        )
    # operator.index rather than isinstance(value, int): a class number read
    # back out of an SCL array is a numpy integer, which is not an int. bool is
    # excluded by hand because it is one, and True is not class 1.
    number = None
    if not isinstance(value, bool):
        try:
            number = operator.index(value)
        except TypeError:
            number = None
    if number is None:
        raise ValidationError(
            f"a scene class must be an SCLClass, a class number 0-11, or a class "
            f"name such as 'cloud_shadows'; got {value!r}"
        )
    try:
        return SCLClass(number)
    except ValueError:
        raise ValidationError(
            f"{number} is not a Sentinel-2 scene class; the classes are numbered "
            f"0-11 ({', '.join(f'{m.value} {m.name.lower()}' for m in SCLClass)})"
        ) from None


def resolve_scl_classes(classes: Iterable[SCLClass | int | str]) -> frozenset[SCLClass]:
    """Resolve an iterable of class numbers or names to :class:`SCLClass` members.

    Parameters
    ----------
    classes : iterable of SCLClass or int or str
        Scene classes, each given as a member, as its number (0-11), or as its
        name, matched case-insensitively (``"cloud_shadows"``). The names ESA
        has renamed are accepted under their older spellings too, so
        ``"bare_soil"`` resolves to :attr:`SCLClass.NOT_VEGETATED` and
        ``"dark_features"`` to :attr:`SCLClass.CAST_SHADOWS`.

    Returns
    -------
    frozenset of SCLClass
        The resolved classes. Duplicates collapse, since naming a class twice
        masks the same pixels once.

    Raises
    ------
    ValidationError
        If ``classes`` is a bare string rather than a sequence of them, if it
        is empty, or if any entry names no scene class. A bare string is
        rejected because iterating one yields characters, so ``"water"`` would
        otherwise be read as five unrelated class names.

    Examples
    --------
    >>> sorted(resolve_scl_classes([3, "thin_cirrus"]))
    [<SCLClass.CLOUD_SHADOWS: 3>, <SCLClass.THIN_CIRRUS: 10>]
    >>> resolve_scl_classes(["bare_soil"])
    frozenset({<SCLClass.NOT_VEGETATED: 5>})
    """
    if isinstance(classes, str):
        raise ValidationError(
            f"expected a sequence of scene classes, such as [3, 8, 9, 10] or "
            f"['cloud_shadows', 'thin_cirrus']; got {classes!r}"
        )
    resolved = frozenset(_resolve_scl_class(value) for value in classes)
    if not resolved:
        raise ValidationError(
            "name at least one scene class to mask; got an empty sequence. To mask "
            "nothing, skip the masking step rather than asking for an empty mask"
        )
    return resolved


def scl_mask(
    scl: np.ndarray, *, classes: Iterable[SCLClass | int | str] | None = None
) -> np.ndarray:
    """Flag the Sentinel-2 pixels belonging to a set of scene classes.

    Reads the ``SCL`` band as what it is — one class number per pixel — and
    reports which pixels fall in the classes being masked. It makes no
    judgement of its own beyond the default class set, and it does not touch
    reflectance data; turning this flag into nodata is the masking operation's
    job.

    Parameters
    ----------
    scl : numpy.ndarray
        The ``SCL`` band, holding one class number per pixel. Any shape and
        any numeric dtype, but the values must have arrived by nearest
        neighbour: an interpolated class number denotes a different class, or
        no class at all.
    classes : iterable of SCLClass or int or str, optional
        Which classes to flag, as members, numbers (0-11), or names. Defaults
        to :data:`SCL_DEFAULT_MASKED` — cloud, cloud shadows, cirrus, and the
        two classes that hold no measurement.

    Returns
    -------
    numpy.ndarray
        A boolean array of ``scl``'s shape, ``True`` where the pixel belongs
        to one of ``classes``. Carries no nodata of its own: a flag is either
        set or it is not, and ``SCL``'s own missing-data class is a class like
        any other.

    Raises
    ------
    ValidationError
        If ``classes`` is a bare string, is empty, or names no scene class.

    Notes
    -----
    Elementwise and array-agnostic: the result is backed by whatever backs
    ``scl``, so a lazy array yields a lazy mask and nothing is materialized
    here.

    Examples
    --------
    >>> import numpy as np
    >>> scl = np.array([[4, 9], [6, 3]], dtype="uint8")
    >>> scl_mask(scl)
    array([[False,  True],
           [False,  True]])
    >>> scl_mask(scl, classes=["water"])
    array([[False, False],
           [ True, False]])
    """
    wanted = SCL_DEFAULT_MASKED if classes is None else resolve_scl_classes(classes)
    # A plain array of the class numbers: np.isin does not accept a set, whose
    # iteration order is unspecified, and sorting keeps the call reproducible.
    values = np.array(sorted(int(member) for member in wanted), dtype="int16")
    return np.isin(scl, values)
