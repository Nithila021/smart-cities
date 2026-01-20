"""
Parse custom demographic group descriptions for personalized safety analysis
"""
import re

def parse_demographic_group(description):
    """
    Parse natural language demographic descriptions
    Examples:
    - "3 white women + 2 black men"
    - "1 queer couple men"
    - "2 asian women, 1 hispanic man"
    - "family with 2 children"
    """
    if not description:
        return None
    
    description = description.lower().strip()
    
    # Initialize group profile
    profile = {
        'total_people': 0,
        'demographics': [],
        'vulnerable_groups': [],
        'risk_factors': [],
        'races': []
    }
    
    # Extract numbers and demographics
    # Pattern: number + optional race/ethnicity + gender/identity
    patterns = [
        r'(\d+)\s*(white|black|asian|hispanic|latino|latina|native|indigenous)?\s*(woman|women|man|men|male|female|person|people|queer|lgbtq|transgender|trans|nonbinary)',
        r'(family|families)\s*with\s*(\d+)?\s*(child|children|kid|kids)',
        r'(\d+)\s*(elderly|senior|seniors)',
        r'(\d+)\s*(teen|teenager|teenagers|youth)',
        r'(couple|couples)\s*(queer|lgbtq|gay|lesbian)?'
    ]
    
    # Count total people
    numbers = re.findall(r'\b(\d+)\b', description)
    if numbers:
        profile['total_people'] = sum(int(n) for n in numbers)
    
    # Extract race/ethnicity
    races = []
    if 'black' in description or 'african' in description:
        races.append('Black')
        profile['risk_factors'].append('racial_bias')
    if 'white' in description or 'caucasian' in description:
        races.append('White')
    if 'asian' in description:
        races.append('Asian')
        profile['risk_factors'].append('racial_bias')
    if 'hispanic' in description or 'latino' in description or 'latina' in description:
        races.append('Hispanic/Latino')
        profile['risk_factors'].append('racial_bias')
    if 'native' in description or 'indigenous' in description:
        races.append('Native American')
        profile['risk_factors'].append('racial_bias')
    
    profile['races'] = races
    
    # Identify vulnerable groups
    if any(word in description for word in ['child', 'children', 'kid', 'kids', 'baby', 'infant']):
        profile['vulnerable_groups'].append('children')
        profile['risk_factors'].append('child_safety')
    
    if any(word in description for word in ['elderly', 'senior', 'old', 'aged']):
        profile['vulnerable_groups'].append('elderly')
        profile['risk_factors'].append('elderly_vulnerability')
    
    if any(word in description for word in ['woman', 'women', 'female', 'girl', 'girls']):
        profile['vulnerable_groups'].append('women')
        profile['risk_factors'].append('gender_based_violence')
    
    if any(word in description for word in ['queer', 'lgbtq', 'gay', 'lesbian', 'transgender', 'trans', 'nonbinary']):
        profile['vulnerable_groups'].append('lgbtq')
        profile['risk_factors'].append('hate_crimes')
    
    if any(word in description for word in ['pregnant', 'pregnancy']):
        profile['vulnerable_groups'].append('pregnant')
        profile['risk_factors'].append('physical_vulnerability')
    
    if any(word in description for word in ['disabled', 'disability', 'wheelchair']):
        profile['vulnerable_groups'].append('disabled')
        profile['risk_factors'].append('accessibility_safety')
    
    # Parse specific demographics
    # Women count
    women_match = re.search(r'(\d+)\s*(?:white|black|asian|hispanic|latino|latina)?\s*(?:woman|women)', description)
    if women_match:
        count = int(women_match.group(1))
        profile['demographics'].append({'type': 'women', 'count': count})
    
    # Men count
    men_match = re.search(r'(\d+)\s*(?:white|black|asian|hispanic|latino)?\s*(?:man|men|male)', description)
    if men_match:
        count = int(men_match.group(1))
        profile['demographics'].append({'type': 'men', 'count': count})
    
    # Children count
    children_match = re.search(r'(\d+)\s*(?:child|children|kid|kids)', description)
    if children_match:
        count = int(children_match.group(1))
        profile['demographics'].append({'type': 'children', 'count': count})
    
    # Set default if no count found
    if profile['total_people'] == 0:
        profile['total_people'] = 1
    
    return profile


def generate_custom_safety_recommendations(profile, safety_score, crime_types):
    """
    Generate personalized safety recommendations based on group composition
    """
    recommendations = []
    
    if not profile:
        return recommendations
    
    # Header - more concise
    recommendations.append(f"\n🔒 CUSTOM GROUP ANALYSIS")
    recommendations.append(f"Group: {profile['total_people']} people")
    
    # Show race if specified
    if profile['races']:
        recommendations.append(f"Race: {', '.join(profile['races'])}")
    
    if profile['vulnerable_groups']:
        recommendations.append(f"Includes: {', '.join(profile['vulnerable_groups']).title()}")
    
    # Risk-specific recommendations - condensed
    if 'children' in profile['vulnerable_groups']:
        if safety_score < 60:
            recommendations.append("\n👶 CHILDREN: ⚠️ HIGH RISK")
            recommendations.append("  • Keep within arm's reach")
            recommendations.append("  • Consider alternative locations")
        else:
            recommendations.append("\n👶 CHILDREN: ✓ Moderate safety")
            recommendations.append("  • Maintain visual contact")
        
        # Check for child-related crimes - only show if found
        child_crimes = ['KIDNAPPING', 'CHILD', 'ABUSE', 'PREDATORY']
        for crime, count in crime_types.items():
            if any(concern in crime.upper() for concern in child_crimes):
                recommendations.append(f"  ⚠️ {crime}: {count} incidents")
    
    if 'women' in profile['vulnerable_groups']:
        if safety_score < 60:
            recommendations.append("\n👩 WOMEN: ⚠️ HEIGHTENED RISK")
            recommendations.append("  • Travel in groups")
            recommendations.append("  • Share live location")
        else:
            recommendations.append("\n👩 WOMEN: ✓ Reasonable safety")
            recommendations.append("  • Stay in populated areas")
        
        # Check for gender-based crimes
        gender_crimes = ['HARASSMENT', 'ASSAULT', 'RAPE', 'SEX']
        for crime, count in crime_types.items():
            if any(concern in crime.upper() for concern in gender_crimes):
                recommendations.append(f"  ⚠️ {crime}: {count} incidents")
    
    if 'lgbtq' in profile['vulnerable_groups']:
        if safety_score < 60:
            recommendations.append("\n🏳️‍🌈 LGBTQ+: ⚠️ CAUTION")
            recommendations.append("  • Stay in LGBTQ+-friendly areas")
            recommendations.append("  • Have support contacts ready")
        else:
            recommendations.append("\n🏳️‍🌈 LGBTQ+: ✓ Generally safe")
            recommendations.append("  • Be aware of surroundings")
        
        # Check for hate crimes
        hate_crimes = ['HATE', 'BIAS', 'HARASSMENT']
        for crime, count in crime_types.items():
            if any(concern in crime.upper() for concern in hate_crimes):
                recommendations.append(f"  ⚠️ {crime}: {count} incidents")
    
    if 'elderly' in profile['vulnerable_groups']:
        if safety_score < 60:
            recommendations.append("\n👴 ELDERLY: ⚠️ HIGH VULNERABILITY")
            recommendations.append("  • Use buddy system")
            recommendations.append("  • Consider rideshare")
        else:
            recommendations.append("\n👴 ELDERLY: ✓ Suitable")
            recommendations.append("  • Ensure good lighting")
    
    # Race-based safety analysis
    if 'racial_bias' in profile['risk_factors'] and profile['races']:
        recommendations.append(f"\n🛡️ RACIAL SAFETY CONSIDERATIONS:")
        
        # Check for hate crimes and bias incidents
        hate_crimes = ['HATE', 'BIAS', 'HARASSMENT']
        bias_found = False
        for crime, count in crime_types.items():
            if any(concern in crime.upper() for concern in hate_crimes):
                recommendations.append(f"  ⚠️ {crime}: {count} incidents")
                bias_found = True
        
        if bias_found or safety_score < 60:
            recommendations.append(f"  • Be aware of surroundings")
            recommendations.append(f"  • Document any incidents")
            recommendations.append(f"  • Know emergency contacts")
        else:
            recommendations.append(f"  ✓ No significant bias incidents reported")
    
    # Group size recommendations - condensed
    if profile['total_people'] == 1:
        recommendations.append("\n⚠️ SOLO: Higher risk")
        recommendations.append("  • Share location")
        recommendations.append("  • Stay in populated areas")
    elif profile['total_people'] >= 5:
        recommendations.append("\n👥 LARGE GROUP: ✓ Safety in numbers")
        recommendations.append("  • Designate leader")
        recommendations.append("  • Don't split up")
    
    return "\n".join(recommendations)
