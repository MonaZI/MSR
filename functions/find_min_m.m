function [m_min] = find_min_m(d)
    p = [1/6 1/3 5/2 -2*d];
    r = roots(p);
    r = r(find(imag(r)==0));
    max_root = max(r);
    if max_root < 0
        m_min = 1;
    else
        m_min = ceil(max_root);
    end
end